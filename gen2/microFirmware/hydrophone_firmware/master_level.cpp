#include "daisy_seed.h"
#include "daisysp.h"
#include "library/fft_library.h"
#include "library/serial_library.h"

using namespace daisy;
using namespace daisysp;
using namespace daisy::seed;

////////////////////////////// Competition Configuration (WE CAN CHANGE)/////////////////////////////////////////
// // Hydrophone normalization (manually calibrate)
// const float hydrophone_0_max = 4.0f;
// const float hydrophone_1_max = 4.0f;

// // FFT
// constexpr size_t kFftSize = 64;             // Higher = better frequency resolution
// constexpr size_t kBlockSize = 64;             // Block size for audio processing

// // RMS
// const float multiplier = 100;                  // Amplification of signal (per sample)

// // Frequency Detection
// const float targetFrequency = 35000.0f;        // Target frequency to detect
// const float frequencyTolerance = 0.01f;       // Tolerance for frequency detection

// // Printing
// constexpr int kPrintIntervalMs = 1;         // Print interval




////////////////////////////// Testing Configuration (WE CAN CHANGE)/////////////////////////////////////////
// Hydrophone normalization (manually calibrate)
const float hydrophone_0_max = 4.0f;
const float hydrophone_1_max = 4.0f;

// FFT
constexpr size_t kFftSize = 64;             // Higher = better frequency resolution
constexpr size_t kBlockSize = 64;             // Block size for audio processing

// RMS
const float multiplier = 100;                  // Amplification of signal (per sample)

// Frequency Detection
const float targetFrequency = 25000.0f;        // Target frequency to detect
const float frequencyTolerance = 0.01f;       // Tolerance for frequency detection

// Printing
constexpr int kPrintIntervalMs = 1;         // Print interval

////////////////////////////// Internal Variables for Master (DO NOT CHANGE) ///////////////////////////////////
// Hardware
DaisySeed hw;

// Global FFT library object
FFTLibrary fftLibrary(96000.f);

// FFT buffers for both microphones (MASTER)
static float DSY_SDRAM_BSS fft_input_buffer_0[kFftSize];
static float DSY_SDRAM_BSS fft_input_buffer_1[kFftSize];
static size_t buffer_write_pos_0 = 0;
static size_t buffer_write_pos_1 = 0;
static bool fft_ready_for_processing_0 = false;
static bool fft_ready_for_processing_1 = false;

// Frequency window (derived)
const float lowerFreq = targetFrequency * (1.0f - frequencyTolerance);
const float upperFreq = targetFrequency * (1.0f + frequencyTolerance);

// Latest raw magnitudes 
float detectedFrequencyLevel_0 = 0.0f;
float detectedFrequencyLevel_1 = 0.0f;

////////////////////////////// Internal Variables for Communication (DO NOT CHANGE) ///////////////////////////////////
// Print pacing
uint32_t lastPrintTime = 0;

// Normalized detected frequency levels (0 ~ 1 = master, 2 ~ 3 = slave)
float normalizedDetectedFrequencyLevel_0 = 0.0f;
float normalizedDetectedFrequencyLevel_1 = 0.0f;
float normalizedDetectedFrequencyLevel_2 = 0.0f;
float normalizedDetectedFrequencyLevel_3 = 0.0f;


////////////////////////////////////////// Setup and Loop //////////////////////////////////////////

void MyCallback(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size) {
    for (size_t i = 0; i < size; i++)
    {
        float sample_0 = in[0][i];
        float sample_1 = in[1][i];
        
        // Amplify signals
        float processedSample_0 = sample_0 * multiplier;  
        float processedSample_1 = sample_1 * multiplier;

        // Fill the FFT buffers if ready for more data
        if (!fft_ready_for_processing_0)
        {
            fft_input_buffer_0[buffer_write_pos_0] = processedSample_0;
            buffer_write_pos_0++;

            if (buffer_write_pos_0 >= kFftSize)
            {
                buffer_write_pos_0 = 0;
                fft_ready_for_processing_0 = true;
            }
        }

        if (!fft_ready_for_processing_1)
        {
            fft_input_buffer_1[buffer_write_pos_1] = processedSample_1;
            buffer_write_pos_1++;

            if (buffer_write_pos_1 >= kFftSize)
            {
                buffer_write_pos_1 = 0;
                fft_ready_for_processing_1 = true;
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

    // Initialize serial
    SerialLibrary serial(hw);
    serial.Init();

    // Start audio
    hw.StartAudio(MyCallback);

    hw.PrintLine("TDOA Frequency Detection Ready");
    hw.PrintLine("Continuous monitoring: printing levels every %d ms", kPrintIntervalMs);

    // Get timestamp
    lastPrintTime = System::GetNow();

    // Initialize ADC on A0 and A1 for slave hydrophones
    AdcChannelConfig adc_cfg[2];
    adc_cfg[0].InitSingle(A0);
    adc_cfg[1].InitSingle(A1);
    hw.adc.Init(adc_cfg, 2);
    hw.adc.Start();

    while (1)
    {
        // Update latest magnitudes when FFT buffers are ready
        if (fft_ready_for_processing_0)
        {
            detectedFrequencyLevel_0 = fftLibrary.getFrequencyMagnitude(fft_input_buffer_0, kFftSize, targetFrequency, frequencyTolerance);
            fft_ready_for_processing_0 = false;
        }

        if (fft_ready_for_processing_1)
        {
            detectedFrequencyLevel_1 = fftLibrary.getFrequencyMagnitude(fft_input_buffer_1, kFftSize, targetFrequency, frequencyTolerance);
            fft_ready_for_processing_1 = false;
        }

        // clip the detected frequency levels
        if (detectedFrequencyLevel_0 > hydrophone_0_max)
        {
            detectedFrequencyLevel_0 = hydrophone_0_max;
        }
        if (detectedFrequencyLevel_1 > hydrophone_1_max)
        {
            detectedFrequencyLevel_1 = hydrophone_1_max;
        }
        
        // Normalize detected frequency levels
        normalizedDetectedFrequencyLevel_0 = detectedFrequencyLevel_0 / hydrophone_0_max;
        normalizedDetectedFrequencyLevel_1 = detectedFrequencyLevel_1 / hydrophone_1_max;

        // Read analog values from A0/A1 (from slave)
        normalizedDetectedFrequencyLevel_2 = hw.adc.GetFloat(0);
        normalizedDetectedFrequencyLevel_3 = hw.adc.GetFloat(1);

        // Periodic print
        uint32_t currentTime = System::GetNow();
        if (currentTime - lastPrintTime >= kPrintIntervalMs)
        {
            // Print raw magnitudes (for calibration)
            // hw.PrintLine("Raw Mic0: " FLT_FMT3 " Raw Mic1: " FLT_FMT3,
            //             FLT_VAR3(detectedFrequencyLevel_0),
            //             FLT_VAR3(detectedFrequencyLevel_1));
        
            hw.PrintLine("hydrophone_log: Mic0 reads" FLT_FMT3 " Mic1 reads" FLT_FMT3 " Mic2 reads" FLT_FMT3 " Mic3 reads" FLT_FMT3,
                        FLT_VAR3(normalizedDetectedFrequencyLevel_0),
                        FLT_VAR3(normalizedDetectedFrequencyLevel_1),
                        FLT_VAR3(normalizedDetectedFrequencyLevel_2),
                        FLT_VAR3(normalizedDetectedFrequencyLevel_3));
            lastPrintTime = currentTime;
        }
    }
} 