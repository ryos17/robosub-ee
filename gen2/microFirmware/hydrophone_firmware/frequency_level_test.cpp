#include "daisy_seed.h"
#include "daisysp.h"
#include "library/fft_library.h"
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace daisy;
using namespace daisysp;


////////////////////////////// Configuration (WE CAN CHANGE)/////////////////////////////////////////
// FFT
constexpr size_t kFftSize = 64;             // Higher = better frequency resolution
constexpr size_t kBlockSize = 64;             // Block size for audio processing
constexpr int kAveragingSamples = 4;          // Average multiple FFT results

// RMS
const float multiplier = 1000;                  // Amplifcation of signal (per sample)

// Frequency Detection
const float targetFrequency = 35000.0f;        // Target frequency to detect
const float frequencyTolerance = 0.00f;       // Tolerance for frequency detection
constexpr int kConsecutiveTriggersNeeded = 2; // Number of consecutive triggers needed to trigger
const float baseThreshold = 0.1f;             // Base threshold for frequency detection

// Printing
constexpr int kPrintIntervalMs = 100;         // Print interval


////////////////////////////// Internal Variables (DO NOT CHANGE) ///////////////////////////////////
// Hardware
DaisySeed hw;

// FFT 
static float DSY_SDRAM_BSS fft_input_buffer[kFftSize];
static size_t buffer_write_pos = 0;
static bool fft_ready_for_processing = false;

// Signal preprocessing  
float frequencyLevelHistory[kAveragingSamples] = {0.0f};
int historyIndex = 0;
float averagedFrequencyLevel = 0.0f;

// RMS amplitude
float cur_rms_amplitude = 0.0f;

// Frequency detection
float detectedFrequencyLevel = 0.0f;
const float lowerFreq = targetFrequency * (1.0f - frequencyTolerance);
const float upperFreq = targetFrequency * (1.0f + frequencyTolerance);
int consecutiveTriggers = 0;
bool isTriggered = false;

// Timing for printing
uint32_t lastPrintTime = 0;

// Sampling rate calculation
uint32_t total_samples;
uint32_t cur_sample_rate;
uint32_t cur_time_ms;
uint32_t prev_time_ms;

// Global FFT library object
FFTLibrary fftLibrary(96000.f); 

////////////////////////////////////////// Setup and Loop //////////////////////////////////////////

void MyCallback(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size) {
    float sum_of_squares = 0.0f;

    for (size_t i = 0; i < size; i++)
    {
        float sample = in[0][i];
        
        // Amplify signal
        float processedSample = sample * multiplier;  
        sum_of_squares += processedSample * processedSample;

        // Fill the FFT buffer if the main loop is ready for more data
        if (!fft_ready_for_processing)
        {
            fft_input_buffer[buffer_write_pos] = processedSample;
            buffer_write_pos++;

            // If the buffer is now full, signal the main loop
            if (buffer_write_pos >= kFftSize)
            {
                buffer_write_pos = 0;
                fft_ready_for_processing = true;
            }
        }
    }

	// RMS calculation
    cur_rms_amplitude = sqrtf(sum_of_squares / size + 1e-9);

	// Actual sample rate calculation
	total_samples += size;
	cur_time_ms = System::GetNow();

	// Once we reach one second, update
	if (cur_time_ms - prev_time_ms >= 1000)
	{
		cur_sample_rate = total_samples;
		total_samples = 0;
		prev_time_ms = cur_time_ms;
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

	// Start program
    hw.StartLog(true);
    hw.StartAudio(MyCallback);

	// Get timestamp
	prev_time_ms = System::GetNow();
	cur_time_ms = System::GetNow();
	lastPrintTime = System::GetNow();

    while (1)
    {
        // If the buffer is full, process the FFT.
        if (fft_ready_for_processing)
        {
            detectedFrequencyLevel = fftLibrary.getFrequencyMagnitude(fft_input_buffer, kFftSize, targetFrequency, frequencyTolerance);
            
            // Signal averaging for noise reduction
            frequencyLevelHistory[historyIndex] = detectedFrequencyLevel;
            historyIndex = (historyIndex + 1) % kAveragingSamples;
            
            // Calculate averaged frequency level
            float sum = 0.0f;
            for (int i = 0; i < kAveragingSamples; ++i) {
                sum += frequencyLevelHistory[i];
            }
            averagedFrequencyLevel = sum / kAveragingSamples;
            
            // Update consecutive trigger count
            if (averagedFrequencyLevel > baseThreshold)
            {
                consecutiveTriggers++;
                if (consecutiveTriggers >= kConsecutiveTriggersNeeded)
                {
                    isTriggered = true;
                }
            }
            else
            {
                // Reset trigger count if threshold not exceeded
                consecutiveTriggers = 0;
                isTriggered = false;
            }
            
            fft_ready_for_processing = false;
        }

        // Print at regular intervals 
        uint32_t currentTime = System::GetNow();
        if (currentTime - lastPrintTime >= kPrintIntervalMs)
        {
            hw.PrintLine("Freq: " FLT_FMT3 " Avg: " FLT_FMT3 " Thresh: " FLT_FMT3 " %s",
                         FLT_VAR3(detectedFrequencyLevel), FLT_VAR3(averagedFrequencyLevel), 
                         FLT_VAR3(baseThreshold), isTriggered ? "TRIGGERED!" : "Below");
            
            lastPrintTime = currentTime;
        }
    }
}