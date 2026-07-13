#include "daisy_seed.h"
#include "daisysp.h"
#include "library/fft_library.h"
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace daisy;
using namespace daisysp;

////////////////////////////// Configuration ///////////////////////////////////
DaisySeed hw;
constexpr int kPrintIntervalMs = 100;

// FFT
constexpr size_t kFftSize = 2048;
constexpr size_t kBlockSize = 64;
static float DSY_SDRAM_BSS fft_input_buffer[kFftSize];
static size_t buffer_write_pos = 0;
static bool fft_ready_for_processing = false;

// RMS amplitude
float cur_rms_amplitude = 0.0f;
const float multiplier = 10000;

// Pitch tracking
float detectedPitch = 0.0f;

// Sampling rate
uint32_t total_samples;
uint32_t cur_sample_rate;
uint32_t cur_time_ms;
uint32_t prev_time_ms;

// Global FFT library object
FFTLibrary fftLibrary(96000.f); // Initialize with default, will be updated

////////////////////////////////////////// Setup and Loop //////////////////////////////////////////

void MyCallback(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size) {
    float sum_of_squares = 0.0f;

    for (size_t i = 0; i < size; i++)
    {
        float sample = in[0][i];
        sum_of_squares += sample * sample;

        // Fill the FFT buffer if the main loop is ready for more data
        if (!fft_ready_for_processing)
        {
            fft_input_buffer[buffer_write_pos] = sample;
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
    cur_rms_amplitude = sqrtf(sum_of_squares / size + 1e-9) * multiplier;

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

    while (1)
    {
        // If the buffer is full, process the FFT.
        if (fft_ready_for_processing)
        {
            detectedPitch = fftLibrary.detectPitch(fft_input_buffer, kFftSize);
            fft_ready_for_processing = false;
        }

        // Print the latest values at a regular interval.
        hw.PrintLine("RMS: " FLT_FMT3 ", Pitch: " FLT_FMT3 " Hz, Sampling rate: %lu Hz",
                     FLT_VAR3(cur_rms_amplitude), FLT_VAR3(detectedPitch), cur_sample_rate);
        System::Delay(kPrintIntervalMs);
    }
}
