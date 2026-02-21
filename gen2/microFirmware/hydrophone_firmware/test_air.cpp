#include "daisy_seed.h"
#include "daisysp.h"
#include "library/fft_library.h"
#include "library/serial_library.h"
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace daisy;
using namespace daisysp;

////////////////////////////// Configuration (WE CAN CHANGE)/////////////////////////////////////////
// // FFT
// constexpr size_t kFftSize = 1024;             // Higher = better frequency resolution
// constexpr size_t kBlockSize = 64;             // Block size for audio processing

// // RMS
// const float multiplier = 100;                  // Amplification of signal (per sample)

// // Frequency Detection
// const float targetFrequency = 35000.0f;        // Target frequency to detect
// const float frequencyTolerance = 0.01f;       // Tolerance for frequency detection
// const float baseThreshold = 1.0f;             // Base threshold for frequency detection

// // Printing
// constexpr int kPrintIntervalMs = 100;         // Print interval
// FFT
constexpr size_t kFftSize = 1024;             // Higher = better frequency resolution
constexpr size_t kBlockSize = 64;             // Block size for audio processing

// RMS
const float multiplier = 100;                  // Amplification of signal (per sample)

// Frequency Detection
const float targetFrequency = 1760.0f;        // Target frequency to detect
const float frequencyTolerance = 0.01f;       // Tolerance for frequency detection
const float baseThreshold = 1.0f;             // Base threshold for frequency detection

// Printing
constexpr int kPrintIntervalMs = 100;         // Print interval

////////////////////////////// Internal Variables (DO NOT CHANGE) ///////////////////////////////////
// Hardware
DaisySeed hw;

// FFT buffers for both microphones
static float DSY_SDRAM_BSS fft_input_buffer_0[kFftSize];
static float DSY_SDRAM_BSS fft_input_buffer_1[kFftSize];
static size_t buffer_write_pos_0 = 0;
static size_t buffer_write_pos_1 = 0;
static bool fft_ready_for_processing_0 = false;
static bool fft_ready_for_processing_1 = false;

// Frequency detection
float detectedFrequencyLevel_0 = 0.0f;
float detectedFrequencyLevel_1 = 0.0f;
const float lowerFreq = targetFrequency * (1.0f - frequencyTolerance);
const float upperFreq = targetFrequency * (1.0f + frequencyTolerance);

// TDOA state
bool waiting_for_start = true;
bool frequency_detected_0 = false;
bool frequency_detected_1 = false;
bool first_buffer_after_start = true;
unsigned long start_time_us = 0;
unsigned long detection_time_0_us = 0;
unsigned long detection_time_1_us = 0;

// Timing for printing
uint32_t lastPrintTime = 0;

// Global FFT library object
FFTLibrary fftLibrary(96000.f);

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

    // Initialize serial communication
    SerialLibrary serial(hw);
    serial.Init();
    
    // Start audio
    hw.StartAudio(MyCallback);

    hw.PrintLine("TDOA Frequency Detection Ready");
    hw.PrintLine("Type 'start' to begin detection...");

    // Get timestamp
    lastPrintTime = System::GetNow();

    while (1)
    {
        // Check for serial input
        if (waiting_for_start)
        {
            if (serial.CheckCommand("start"))
            {
                                 waiting_for_start = false;
                 start_time_us = System::GetUs();
                 frequency_detected_0 = false;
                 frequency_detected_1 = false;
                 first_buffer_after_start = true;
                
                                 // Reset FFT buffers to clear any old data
                 buffer_write_pos_0 = 0;
                 buffer_write_pos_1 = 0;
                 fft_ready_for_processing_0 = false;
                 fft_ready_for_processing_1 = false;
                 
                                  // Clear the actual buffer contents
                 for (size_t i = 0; i < kFftSize; i++) {
                     fft_input_buffer_0[i] = 0.0f;
                     fft_input_buffer_1[i] = 0.0f;
                 }
                 
                 // Wait a moment to ensure we start with fresh data
                 System::Delay(10);
                 
                 hw.PrintLine("Starting TDOA detection... (start_time: %lu)", start_time_us);
            }
            else
            {
                // Print frequency levels while waiting for start command
                uint32_t currentTime = System::GetNow();
                if (currentTime - lastPrintTime >= kPrintIntervalMs)
                {
                    // Process FFT for both microphones to get current levels
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
                    
                    hw.PrintLine("Mic0: " FLT_FMT3 " Mic1: " FLT_FMT3, FLT_VAR3(detectedFrequencyLevel_0), FLT_VAR3(detectedFrequencyLevel_1));
                    
                    lastPrintTime = currentTime;
                }
            }
            continue;
        }

        // Process FFT for microphone 0
        if (fft_ready_for_processing_0 && !frequency_detected_0)
        {
            // Skip the first buffer after start to ensure fresh data
            if (first_buffer_after_start) {
                 first_buffer_after_start = false;
                 fft_ready_for_processing_0 = false;
                 continue;
            }
             
            detectedFrequencyLevel_0 = fftLibrary.getFrequencyMagnitude(fft_input_buffer_0, kFftSize, targetFrequency, frequencyTolerance);
             
            if (detectedFrequencyLevel_0 > baseThreshold)
            {
                frequency_detected_0 = true;
                detection_time_0_us = System::GetUs();
                // Handle potential timer overflow
                unsigned long time_diff;
                if (detection_time_0_us >= start_time_us) {
                    time_diff = detection_time_0_us - start_time_us;
                } else {
                    // Timer overflow occurred
                    time_diff = (0xFFFFFFFF - start_time_us) + detection_time_0_us + 1;
                }
                hw.PrintLine("Frequency detected on mic 0 at %lu μs (level: " FLT_FMT3 ") [start:%lu, detect:%lu]", time_diff, FLT_VAR3(detectedFrequencyLevel_0), start_time_us, detection_time_0_us);
            }
             
            fft_ready_for_processing_0 = false;
         }

                 // Process FFT for microphone 1
         if (fft_ready_for_processing_1 && !frequency_detected_1)
         {
             // Skip the first buffer after start to ensure fresh data
             if (first_buffer_after_start) {
                 fft_ready_for_processing_1 = false;
                 continue;
             }
             
             detectedFrequencyLevel_1 = fftLibrary.getFrequencyMagnitude(fft_input_buffer_1, kFftSize, targetFrequency, frequencyTolerance);
             
            if (detectedFrequencyLevel_1 > baseThreshold)
              {
                  frequency_detected_1 = true;
                  detection_time_1_us = System::GetUs();
                  // Handle potential timer overflow
                  unsigned long time_diff;
                  if (detection_time_1_us >= start_time_us) {
                      time_diff = detection_time_1_us - start_time_us;
                  } else {
                      // Timer overflow occurred
                      time_diff = (0xFFFFFFFF - start_time_us) + detection_time_1_us + 1;
                  }
                  hw.PrintLine("Frequency detected on mic 1 at %lu μs (level: " FLT_FMT3 ") [start:%lu, detect:%lu]", time_diff, FLT_VAR3(detectedFrequencyLevel_1), start_time_us, detection_time_1_us);
              }
             
             fft_ready_for_processing_1 = false;
         }

        // If both microphones detected the frequency, calculate TDOA
        if (frequency_detected_0 && frequency_detected_1)
        {
            // Calculate TDOA as absolute difference between detection times
            unsigned long time_diff_us;
            if (detection_time_1_us >= detection_time_0_us) {
                time_diff_us = detection_time_1_us - detection_time_0_us;
            } else {
                time_diff_us = detection_time_0_us - detection_time_1_us;
            }
            
            // Determine which microphone detected first
            if (detection_time_1_us < detection_time_0_us) {
                hw.PrintLine("TDOA: %lu μs (Mic 1 detected first)", time_diff_us);
            } else {
                hw.PrintLine("TDOA: %lu μs (Mic 0 detected first)", time_diff_us);
            }
            hw.PrintLine("Ready for next measurement. Type 'start' to begin...");
            
            // Reset for next measurement
            waiting_for_start = true;
            frequency_detected_0 = false;
            frequency_detected_1 = false;

            System::Delay(5000);
        }
    }
} 