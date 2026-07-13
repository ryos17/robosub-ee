#pragma once

#include <complex>
#include <vector>

class FFTLibrary
{
public:
    FFTLibrary(float sampleRate);
    
    // Main FFT function
    void fft(std::vector<std::complex<float>>& signal);
    
    // Pitch detection function
    float detectPitch(const float* audio_buffer, size_t buffer_size);
    
    // Level detection function - get magnitude at specific frequency with tolerance
    float getFrequencyMagnitude(const float* audio_buffer, size_t buffer_size, float target_freq, float tolerance = 0.05f);
    
    // Utility functions
    static void applyHanningWindow(std::vector<std::complex<float>>& signal);
    static float findInterpolatedFrequency(const std::vector<std::complex<float>>& fft_data, 
                                         float sample_rate);

private:
    float m_sampleRate;
}; 