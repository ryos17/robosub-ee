#include "daisy_seed.h"
#include "daisysp.h"
#include "library/fft_library.h"
#include "library/serial_library.h"

using namespace daisy;
using namespace daisysp;

////////////////////////////// Competition Configuration (WE CAN CHANGE)/////////////////////////////////////////
// // Hydrophone normalization (manually calibrate)
// const float hydrophone_2_max = 3.5f;
// const float hydrophone_3_max = 3.5f;

// // FFT
// constexpr size_t kFftSize = 64;             // Higher = better frequency resolution
// constexpr size_t kBlockSize = 64;             // Block size for audio processing

// // RMS
// const float multiplier = 100;                  // Amplification of signal (per sample)

// // Frequency Detection
// const float targetFrequency = 25000.0f;        // Target frequency to detect
// const float frequencyTolerance = 0.01f;       // Tolerance for frequency detection




// ////////////////////////////// Testing Configuration (WE CAN CHANGE)/////////////////////////////////////////
// Hydrophone normalization (manually calibrate)
const float hydrophone_2_max = 4.0f;
const float hydrophone_3_max = 4.0f;

// FFT
constexpr size_t kFftSize = 64;             // Higher = better frequency resolution
constexpr size_t kBlockSize = 64;             // Block size for audio processing

// RMS
const float multiplier = 100;                  // Amplification of signal (per sample)

// Frequency Detection
const float targetFrequency = 25000.0f;        // Target frequency to detect
const float frequencyTolerance = 0.01f;       // Tolerance for frequency detection

////////////////////////////// Internal Variables for Master (DO NOT CHANGE) ///////////////////////////////////
// Hardware
static DaisySeed hw;

// Global FFT library object
FFTLibrary fftLibrary(96000.f);

// FFT buffers for both microphones (SLAVE channels 2 and 3)
static float DSY_SDRAM_BSS fft_input_buffer_2[kFftSize];
static float DSY_SDRAM_BSS fft_input_buffer_3[kFftSize];
static size_t buffer_write_pos_2 = 0;
static size_t buffer_write_pos_3 = 0;
static bool fft_ready_for_processing_2 = false;
static bool fft_ready_for_processing_3 = false;

// Frequency window (derived)
const float lowerFreq = targetFrequency * (1.0f - frequencyTolerance);
const float upperFreq = targetFrequency * (1.0f + frequencyTolerance);

// Latest magnitudes
float detectedFrequencyLevel_2 = 0.0f;
float detectedFrequencyLevel_3 = 0.0f;

// Converted magnitudes (0 --> 0V, 4095 --> 3.3V)
uint16_t uintDetectedFrequencyLevel_2 = 0;
uint16_t uintDetectedFrequencyLevel_3 = 0;

////////////////////////////////////////// Setup and Loop //////////////////////////////////////////

uint16_t map(float x, float in_min, float in_max, int out_min, int out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

void MyCallback(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size) {
    for (size_t i = 0; i < size; i++)
    {
        float sample_2 = in[0][i];
        float sample_3 = in[1][i];
        
        // Amplify signals
        float processedSample_2 = sample_2 * multiplier;  
        float processedSample_3 = sample_3 * multiplier;

        // Fill the FFT buffers if ready for more data
        if (!fft_ready_for_processing_2)
        {
            fft_input_buffer_2[buffer_write_pos_2] = processedSample_2;
            buffer_write_pos_2++;

            if (buffer_write_pos_2 >= kFftSize)
            {
                buffer_write_pos_2 = 0;
                fft_ready_for_processing_2 = true;
            }
        }

        if (!fft_ready_for_processing_3)
        {
            fft_input_buffer_3[buffer_write_pos_3] = processedSample_3;
            buffer_write_pos_3++;

            if (buffer_write_pos_3 >= kFftSize)
            {
                buffer_write_pos_3 = 0;
                fft_ready_for_processing_3 = true;
            }
        }
    }
}

int main(void)
{
    // Initialize the Daisy Seed Hardware
    hw.Init();
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_96KHZ);
    hw.SetAudioBlockSize(kBlockSize);

    // Initialize the FFT library with the actual sample rate
    fftLibrary = FFTLibrary(hw.AudioSampleRate());

    // // Initialize serial
    // SerialLibrary serial(hw);
    // serial.Init();

    // Initialize DAC outputs
	DacHandle::Config cfg;
	cfg.bitdepth   = DacHandle::BitDepth::BITS_12;
	cfg.buff_state = DacHandle::BufferState::ENABLED;
	cfg.mode       = DacHandle::Mode::POLLING;
	cfg.chn        = DacHandle::Channel::BOTH;
	hw.dac.Init(cfg);
	hw.dac.WriteValue(DacHandle::Channel::BOTH, 0);
	hw.dac.WriteValue(DacHandle::Channel::ONE, 0); 
	hw.dac.WriteValue(DacHandle::Channel::TWO, 0); 

    System::Delay(100);

    // Start audio (after DAC is initialized)
    hw.StartAudio(MyCallback);

    while (1)
    {
        // Update latest magnitudes when FFT buffers are ready
        if (fft_ready_for_processing_2)
        {
            detectedFrequencyLevel_2 = fftLibrary.getFrequencyMagnitude(fft_input_buffer_2, kFftSize, targetFrequency, frequencyTolerance);
            fft_ready_for_processing_2 = false;
        }

        if (fft_ready_for_processing_3)
        {
            detectedFrequencyLevel_3 = fftLibrary.getFrequencyMagnitude(fft_input_buffer_3, kFftSize, targetFrequency, frequencyTolerance);
            fft_ready_for_processing_3 = false;
        }
        
        // clip the detected frequency levels
        if (detectedFrequencyLevel_2 > hydrophone_2_max)
        {
            detectedFrequencyLevel_2 = hydrophone_2_max;
        }
        if (detectedFrequencyLevel_3 > hydrophone_3_max)
        {
            detectedFrequencyLevel_3 = hydrophone_3_max;
        }

        // Map to 0~3.3V range for DAC (0 --> 0V, 4095 --> 3.3V)
        uintDetectedFrequencyLevel_2 = map(detectedFrequencyLevel_2, 0.0f, hydrophone_2_max, 0, 4095);
        uintDetectedFrequencyLevel_3 = map(detectedFrequencyLevel_3, 0.0f, hydrophone_3_max, 0, 4095);
        
        // // Print raw microphone values (for normalizing microphone levels)
        // hw.PrintLine("Raw Mic2: " FLT_FMT3 " Raw Mic3: " FLT_FMT3,
        //             FLT_VAR3(detectedFrequencyLevel_2),
        //             FLT_VAR3(detectedFrequencyLevel_3));
        // System::Delay(100);

        hw.dac.WriteValue(DacHandle::Channel::ONE, uintDetectedFrequencyLevel_2);
        hw.dac.WriteValue(DacHandle::Channel::TWO, uintDetectedFrequencyLevel_3);
    }
} 