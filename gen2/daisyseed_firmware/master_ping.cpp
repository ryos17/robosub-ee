#include "daisy_seed.h"
#include "daisysp.h"
#include "library/fft_library.h"
#include "library/serial_library.h"
#include <algorithm>

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
// const float targetFrequency = 25000.0f;        // Target frequency to detect
// const float frequencyTolerance = 0.01f;       // Tolerance for frequency detection
// const float baseThreshold = 0.04f;             // Base threshold for frequency detection

// // Ping Detection
// const uint32_t listenTimeMs = 10000;          // Duration of listening for ping (ms)
// const uint32_t offThresholdMs = 1000;         // Threshold for off-time detection (ms)
// const uint32_t withinThresholdUs = 3000;      // Threshold for within-time detection (us)


////////////////////////////// Testing Configuration (WE CAN CHANGE)/////////////////////////////////////////
// Hydrophone normalization (manually calibrate)
const float hydrophone_0_max = 4.0f;
const float hydrophone_1_max = 4.0f;

// FFT
constexpr size_t kFftSize = 64;             // Higher = better frequency resolution
constexpr size_t kBlockSize = 64;             // Block size for audio processing

// RMS
const float multiplier = 100;                 // Amplification of signal (per sample)

// Frequency Detection
const float targetFrequency = 14080.0f;        // Target frequency to detect
const float frequencyTolerance = 0.01f;       // Tolerance for frequency detection
const float baseThreshold = 0.02f;             // Base threshold for frequency detection

// Ping Detection
const uint32_t listenTimeMs = 10000;          // Duration of listening for ping (ms)
const uint32_t offThresholdMs = 1000;         // Threshold for off-time detection (ms)
const uint32_t withinThresholdUs = 1000000;      // Threshold for within-time detection (us)

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
// Normalized detected frequency levels (0 ~ 1 = master, 2 ~ 3 = slave)
float normalizedDetectedFrequencyLevel_0 = 0.0f;
float normalizedDetectedFrequencyLevel_1 = 0.0f;
float normalizedDetectedFrequencyLevel_2 = 0.0f;
float normalizedDetectedFrequencyLevel_3 = 0.0f;

// Threshold crossing state and start time (us)
bool wasAboveThreshold_0 = false;
bool wasAboveThreshold_1 = false;
bool wasAboveThreshold_2 = false;
bool wasAboveThreshold_3 = false;


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

    // Initialize ADC on A0 and A1 for slave hydrophones
    AdcChannelConfig adc_cfg[2];
    adc_cfg[0].InitSingle(A0);
    adc_cfg[1].InitSingle(A1);
    hw.adc.Init(adc_cfg, 2);
    hw.adc.Start();

    while (1)
    {
        if (serial.CheckCommand("ping"))
        {
            hw.PrintLine("localization for " FLT_FMT3 " Hz starting!! (wait %lu ms)", FLT_VAR3(targetFrequency), listenTimeMs);

            // Variables for ping detection
            int front_counter = 0;
            int back_counter = 0;
            uint32_t startTimeMs = System::GetNow();
            uint32_t currentTimeMs = startTimeMs;
            uint32_t mostRecentPingTimeMs = startTimeMs;
            bool canBeMeasured = false;
            std::uint32_t recievedTimeUs[4] = {0, 0, 0, 0};

            // Localization for 10 seconds
            while (currentTimeMs - startTimeMs <= listenTimeMs)
            {
                // FFT
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

                // Threshold detection
                if (detectedFrequencyLevel_0 > hydrophone_0_max)
                {
                    detectedFrequencyLevel_0 = hydrophone_0_max;
                }
                if (detectedFrequencyLevel_1 > hydrophone_1_max)
                {
                    detectedFrequencyLevel_1 = hydrophone_1_max;
                }
                normalizedDetectedFrequencyLevel_0 = detectedFrequencyLevel_0 / hydrophone_0_max;
                normalizedDetectedFrequencyLevel_1 = detectedFrequencyLevel_1 / hydrophone_1_max;
                normalizedDetectedFrequencyLevel_2 = hw.adc.GetFloat(0);
                normalizedDetectedFrequencyLevel_3 = hw.adc.GetFloat(1);
                bool isAbove_0 = normalizedDetectedFrequencyLevel_0 >= baseThreshold;
                bool isAbove_1 = normalizedDetectedFrequencyLevel_1 >= baseThreshold;
                bool isAbove_2 = normalizedDetectedFrequencyLevel_2 >= baseThreshold;
                bool isAbove_3 = normalizedDetectedFrequencyLevel_3 >= baseThreshold;
                if (isAbove_0 && !wasAboveThreshold_0)
                {
                    if (canBeMeasured) 
                    {
                        recievedTimeUs[0] = System::GetUs();
                    }
                    // hw.PrintLine("Hydrophone 0 recieved");
                    mostRecentPingTimeMs = System::GetNow();
                }
                if (isAbove_1 && !wasAboveThreshold_1)
                {
                    if (canBeMeasured) 
                    {
                        recievedTimeUs[1] = System::GetUs();
                    }
                    //hw.PrintLine("Hydrophone 1 recieved");
                    mostRecentPingTimeMs = System::GetNow();
                }
                if (isAbove_2 && !wasAboveThreshold_2)
                {
                    if (canBeMeasured) 
                    {
                        recievedTimeUs[2] = System::GetUs();
                    }
                    //hw.PrintLine("Hydrophone 2 recieved");
                    mostRecentPingTimeMs = System::GetNow();
                }
                if (isAbove_3 && !wasAboveThreshold_3)
                {
                    if (canBeMeasured) 
                    {
                        recievedTimeUs[3] = System::GetUs();
                    }
                    //hw.PrintLine("Hydrophone 3 recieved");
                    mostRecentPingTimeMs = System::GetNow();
                }
                wasAboveThreshold_0 = isAbove_0;
                wasAboveThreshold_1 = isAbove_1;
                wasAboveThreshold_2 = isAbove_2;
                wasAboveThreshold_3 = isAbove_3;

                // Once all the pingers are recieved, we can measure the TDOA
                if (recievedTimeUs[0] != 0 && recievedTimeUs[1] != 0 && recievedTimeUs[2] != 0 && recievedTimeUs[3] != 0)
                {
                    //hw.PrintLine("Hydrophones received: %lu, %lu, %lu, %lu us", recievedTimeUs[0], recievedTimeUs[1], recievedTimeUs[2], recievedTimeUs[3]);
                    // Make sure the largest time difference is less than the within threshold
                    uint32_t latest = recievedTimeUs[0];
                    uint32_t earliest = recievedTimeUs[0];
                    for (int i = 1; i < 4; ++i)
                    {
                        if (recievedTimeUs[i] > latest)   { latest = recievedTimeUs[i]; }
                        if (recievedTimeUs[i] < earliest) { earliest = recievedTimeUs[i]; }
                    }
                    if (latest - earliest < withinThresholdUs)
                    {
                        // hw.PrintLine("Measurement is valid");
                        // Find indices of two smallest values using std::min
                        int smallest_idx = 0;
                        for (int i = 1; i < 4; i++) {
                            if (recievedTimeUs[i] < recievedTimeUs[smallest_idx]) {
                                smallest_idx = i;
                            }
                        }
                        int second_smallest_idx = (smallest_idx == 0) ? 1 : 0;
                        for (int i = 0; i < 4; i++) {
                            if (i != smallest_idx && recievedTimeUs[i] < recievedTimeUs[second_smallest_idx]) {
                                second_smallest_idx = i;
                            }
                        }

                        // Update the front and back counter
                        if (smallest_idx == 0  || smallest_idx == 2)
                        {
                            hw.PrintLine("front detected");
                            front_counter++;
                        }
                        else
                        {
                            hw.PrintLine("back detected");
                            back_counter++;
                        }
                        if (second_smallest_idx == 0  || second_smallest_idx == 2)
                        {
                            hw.PrintLine("front detected");
                            front_counter++;
                        }
                        else
                        {
                            hw.PrintLine("back detected");
                            back_counter++;
                        }
                    }
                    // Reset the recieved time
                    recievedTimeUs[0] = 0;
                    recievedTimeUs[1] = 0;
                    recievedTimeUs[2] = 0;
                    recievedTimeUs[3] = 0;
                    canBeMeasured = false;
                }

                // Update current time
                currentTimeMs = System::GetNow();

                // See if the ping is okay to be detected
                if (currentTimeMs - mostRecentPingTimeMs >= offThresholdMs) {
                    canBeMeasured = true;
                }
            }
            if (front_counter > back_counter)
            {
                hw.PrintLine("hydrophone:front");
            }
            else if (front_counter < back_counter)
            {
                hw.PrintLine("hydrophone:back");
            } 
            else if (front_counter == 0 && back_counter == 0)
            {
                hw.PrintLine("hydrophone:no valid ping detected");
            } 
            else 
            {
                hw.PrintLine("hydrophone:inconclusive");
            }
        }
    }
} 