#include "daisy_seed.h"
#include "library/fft_library.h"
#include "library/serial_library.h"
#include <algorithm>

using namespace daisy;
using namespace daisy::seed;

// ////////////////////////////// Competition Configuration (WE CAN CHANGE)/////////////////////////////////////////
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

// // Ping Detection (pinger fires a few ms every ~2 s, periodic)
// const uint32_t offThresholdMs = 1000;         // Silence gap (ms) that re-arms a measurement
// const uint32_t withinThresholdUs = 3000;      // Max spread (us) across the 4 hydrophones for a valid ping

////////////////////////////// Competition Configuration (WE CAN CHANGE)/////////////////////////////////////////
// Hydrophone normalization (manually calibrate)
const float hydrophone_0_max = 4.0f;
const float hydrophone_1_max = 4.0f;

// FFT
constexpr size_t kFftSize = 64;             // Higher = better frequency resolution
constexpr size_t kBlockSize = 64;             // Block size for audio processing

// RMS
const float multiplier = 100;                  // Amplification of signal (per sample)

// Frequency Detection
const float targetFrequency = 1046.0f;        // Target frequency to detect
const float frequencyTolerance = 0.01f;       // Tolerance for frequency detection
const float baseThreshold = 0.1f;             // Base threshold for frequency detection

// Ping Detection (pinger fires a few ms every ~2 s, periodic)
const uint32_t offThresholdMs = 1000;         // Silence gap (ms) that re-arms a measurement
const uint32_t withinThresholdUs = 3000;      // Collection-window length (us) after the first detection

// Output
constexpr uint32_t kPrintIntervalMs = 200;    // Sticky front/back is printed every this many ms

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

// Returns true if a hydrophone index is on the front of the array.
// Wiring: front = hydrophones 0 and 3, back = hydrophones 1 and 2
// (0/1 = master codec inputs, 2/3 = slave).
static inline bool isFront(int idx)
{
    return idx == 0 || idx == 3;
}

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

    // Initialize serial without blocking so the board boots even if the ROS
    // host has not opened the port yet. serial.Poll() below keeps it reflashable
    // ("reboot" -> DFU) via `make flash`.
    SerialLibrary serial(hw);
    serial.Init(false);

    // Start audio
    hw.StartAudio(MyCallback);

    // Initialize ADC on A0 and A1 for slave hydrophones
    AdcChannelConfig adc_cfg[2];
    adc_cfg[0].InitSingle(A0);
    adc_cfg[1].InitSingle(A1);
    hw.adc.Init(adc_cfg, 2);
    hw.adc.Start();

    // ---- Continuous sticky front/back detection ----
    // The pinger fires the array for a few ms every ~2 s. The FIRST hydrophone
    // to cross threshold starts a measurement; over a short collection window we
    // record which hydrophones fire and in what order, then decide direction:
    //   direction = majority of front{0,3} vs back{1,2} among the hydrophones
    //               that fired during this ping; a tie is broken by the earliest
    //               (first) arrival.
    // Examples (front={0,3}, back={1,2}):
    //   2 1 0 3 -> back  (front 2 vs back 2, tie -> earliest 2 = back)
    //   2 0 3   -> front (front 2 vs back 1)
    //   2       -> back  (earliest/only arrival is a back hydrophone)
    // The decision is sticky: it holds until a later ping flips it, and is
    // printed every kPrintIntervalMs.
    bool directionFront = false;      // latched decision (false = back); arbitrary until the first ping
    bool measuring      = false;      // currently inside a collection window
    bool armed          = false;      // ready to start a new measurement (armed after the array is quiet)
    uint32_t collectStartUs = 0;      // us of the first detection in the current ping
    bool fired[4]    = {false, false, false, false};
    int  orderSeq[4] = {-1, -1, -1, -1};   // hydrophone indices in order of arrival
    int  orderCount  = 0;
    bool wasAbove[4] = {false, false, false, false};
    uint32_t lastCrossMs = System::GetNow();   // last threshold crossing (for re-arm)
    uint32_t lastPrintMs = System::GetNow();

    while (1)
    {
        // Handle "reboot" (DFU for make flash) from the host
        serial.Poll();

        // FFT for the master's own hydrophones (0/1)
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
        if (detectedFrequencyLevel_0 > hydrophone_0_max) { detectedFrequencyLevel_0 = hydrophone_0_max; }
        if (detectedFrequencyLevel_1 > hydrophone_1_max) { detectedFrequencyLevel_1 = hydrophone_1_max; }

        // Normalized levels: 0/1 from this board's FFT, 2/3 from the slave via ADC
        normalizedDetectedFrequencyLevel_0 = detectedFrequencyLevel_0 / hydrophone_0_max;
        normalizedDetectedFrequencyLevel_1 = detectedFrequencyLevel_1 / hydrophone_1_max;
        normalizedDetectedFrequencyLevel_2 = hw.adc.GetFloat(0);
        normalizedDetectedFrequencyLevel_3 = hw.adc.GetFloat(1);
        bool isAbove[4] = {
            normalizedDetectedFrequencyLevel_0 >= baseThreshold,
            normalizedDetectedFrequencyLevel_1 >= baseThreshold,
            normalizedDetectedFrequencyLevel_2 >= baseThreshold,
            normalizedDetectedFrequencyLevel_3 >= baseThreshold,
        };

        // Rising-edge detection per hydrophone; drives the measurement state machine
        for (int i = 0; i < 4; i++)
        {
            bool rising = isAbove[i] && !wasAbove[i];
            if (rising)
            {
                lastCrossMs = System::GetNow();
                if (armed && !measuring)
                {
                    // First detection of a new ping -> start collecting arrival order
                    measuring = true;
                    armed = false;
                    collectStartUs = System::GetUs();
                    for (int k = 0; k < 4; k++) { fired[k] = false; orderSeq[k] = -1; }
                    orderCount = 0;
                }
                if (measuring && !fired[i])
                {
                    fired[i] = true;
                    orderSeq[orderCount++] = i;
                }
            }
            wasAbove[i] = isAbove[i];
        }

        // Close the collection window and decide direction
        if (measuring && (System::GetUs() - collectStartUs >= withinThresholdUs))
        {
            int frontCount = 0, backCount = 0;
            for (int i = 0; i < 4; i++)
            {
                if (fired[i]) { if (isFront(i)) { frontCount++; } else { backCount++; } }
            }
            if (frontCount > backCount)       { directionFront = true;  }
            else if (backCount > frontCount)  { directionFront = false; }
            else                              { directionFront = isFront(orderSeq[0]); }  // tie -> earliest arrival
            measuring = false;   // stay disarmed until the array is quiet again
        }

        // Re-arm for the next ping once the array has been quiet long enough
        if (!measuring && !armed && (System::GetNow() - lastCrossMs >= offThresholdMs))
        {
            armed = true;
        }

        // Sticky output: print the latched direction every kPrintIntervalMs
        if (System::GetNow() - lastPrintMs >= kPrintIntervalMs)
        {
            hw.PrintLine(directionFront ? "front" : "back");
            lastPrintMs = System::GetNow();
        }
    }
}
