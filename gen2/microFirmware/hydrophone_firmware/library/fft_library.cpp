#include "fft_library.h"
#include <cmath>

FFTLibrary::FFTLibrary(float sampleRate) : m_sampleRate(sampleRate) {}

// Helper function for parabolic interpolation
float FFTLibrary::findInterpolatedFrequency(const std::vector<std::complex<float>>& fft_data, float sample_rate)
{
    const size_t N = fft_data.size();
    float max_mag = 0.0f;
    size_t max_index = 0;

    // Find the index with the highest magnitude in the first half of the FFT output
    for (size_t i = 1; i < N / 2 - 1; ++i)
    { 
        float magnitude = std::norm(fft_data[i]);

        // Avoid boundary issues
        if (magnitude > max_mag)
        {
            max_mag = magnitude;
            max_index = i;
        }
    }

    // Check if the peak is valid
    if (max_index < 1 || max_index >= N / 2 - 1)
    {
        return (float)max_index * sample_rate / N;
    }

    // Get magnitudes of the peak and its two neighbors
    float mag0 = std::sqrt(std::norm(fft_data[max_index - 1]));
    float mag1 = std::sqrt(std::norm(fft_data[max_index]));
    float mag2 = std::sqrt(std::norm(fft_data[max_index + 1]));

    // Parabolic interpolation formula to find the shift from the bin's center frequency
    float peak_shift = 0.5f * (mag0 - mag2) / (mag0 - 2.0f * mag1 + mag2);
    
    // If the denominator is zero, return the non-interpolated freq.
    if (std::isnan(peak_shift)) {
        return (float)max_index * sample_rate / N;
    }

    // Final calculated frequency
    float frequency = ((float)max_index + peak_shift) * sample_rate / N;

    return frequency;
}

// Apply Hanning window
void FFTLibrary::applyHanningWindow(std::vector<std::complex<float>>& signal)
{
    const size_t N = signal.size();
    for (size_t i = 0; i < N; ++i)
    {
        float window_value = 0.5f * (1.0f - cosf((float)(2.0 * M_PI * i / (N - 1))));
        signal[i] *= window_value;
    }
}

// Recursive Cooley-Tukey Radix-2 FFT implementation.
void FFTLibrary::fft(std::vector<std::complex<float>>& signal)
{
    const size_t N = signal.size();
    if (N <= 1)
        return;

    // Split into even and odd parts
    std::vector<std::complex<float>> even(N / 2);
    std::vector<std::complex<float>> odd(N / 2);
    for (size_t i = 0; i < N / 2; ++i)
    {
        even[i] = signal[2 * i];
        odd[i]  = signal[2 * i + 1];
    }

    // Recurse
    fft(even);
    fft(odd);

    // Combine
    for (size_t k = 0; k < N / 2; ++k)
    {
        std::complex<float> t = std::polar(1.0f, (float)(-2.0 * M_PI * k / N)) * odd[k];
        signal[k]           = even[k] + t;
        signal[k + N / 2]   = even[k] - t;
    }
}

// Pitch detection function
float FFTLibrary::detectPitch(const float* audio_buffer, size_t buffer_size)
{
    // Create a complex vector for the FFT
    std::vector<std::complex<float>> fftSignal(buffer_size, {0.0f, 0.0f});

    // Copy the audio data from our input buffer into the complex vector
    for (size_t i = 0; i < buffer_size; ++i)
    {
        fftSignal[i] = std::complex<float>(audio_buffer[i], 0.0f);
    }

    applyHanningWindow(fftSignal);
    fft(fftSignal);
    
    // Find the fundamental frequency using interpolation
    return findInterpolatedFrequency(fftSignal, m_sampleRate);
}

// Level detection function - get magnitude at specific frequency with tolerance
float FFTLibrary::getFrequencyMagnitude(const float* audio_buffer, size_t buffer_size, float target_freq, float tolerance)
{
    // Create a complex vector for the FFT
    std::vector<std::complex<float>> fftSignal(buffer_size, {0.0f, 0.0f});

    // Copy the audio data from our input buffer into the complex vector
    for (size_t i = 0; i < buffer_size; ++i)
    {
        fftSignal[i] = std::complex<float>(audio_buffer[i], 0.0f);
    }

    applyHanningWindow(fftSignal);
    fft(fftSignal);
    
    // Calculate frequency bounds with tolerance
    float lower_freq = target_freq * (1.0f - tolerance);
    float upper_freq = target_freq * (1.0f + tolerance);
    
    // Calculate bin indices for the frequency range
    size_t lower_bin = (size_t)(lower_freq * buffer_size / m_sampleRate);
    size_t upper_bin = (size_t)(upper_freq * buffer_size / m_sampleRate);
    
    // Ensure we're within valid range (first half of FFT)
    if (lower_bin >= buffer_size / 2)
    {
        lower_bin = buffer_size / 2 - 1;
    }
    if (upper_bin >= buffer_size / 2)
    {
        upper_bin = buffer_size / 2 - 1;
    }
    
    // Sum the magnitudes within the frequency range
    float total_magnitude = 0.0f;
    for (size_t bin = lower_bin; bin <= upper_bin; ++bin)
    {
        float magnitude = std::sqrt(std::norm(fftSignal[bin]));
        total_magnitude += magnitude;
    }
    
    return total_magnitude;
} 