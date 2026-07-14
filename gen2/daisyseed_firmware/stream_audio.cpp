#include "daisy_seed.h"
#include "library/serial_library.h"

using namespace daisy;

// Streams this board's two codec inputs (hydrophones) to the host as raw PCM
// over USB CDC, decimated 96 kHz -> 16 kHz for speech/Whisper use.
// Board-agnostic: flash it on master (hydrophones 0/1) or slave (2/3).
// Host side: research/whisper_ivc/stream_transcribe.py on the Orin.
//
// Wire format, little-endian, 132-byte frames:
//   [0xDA][0x7A][seq u8][reserved u8][32 x (ch0 s16, ch1 s16)]

////////////////////////////// Configuration (WE CAN CHANGE) /////////////////
constexpr size_t kBlockSize  = 48;       // audio callback block size
constexpr int    kDecimation = 6;        // 96 kHz / 6 = 16 kHz out
const float      kGain       = 100.0f;   // same input gain as level firmware

////////////////////////////// Internal (DO NOT CHANGE) //////////////////////
DaisySeed hw;

constexpr size_t kSamplesPerFrame = 32;  // per channel, per USB frame
constexpr size_t kHeaderSize      = 4;
constexpr size_t kFrameSize = kHeaderSize + kSamplesPerFrame * 2 * sizeof(int16_t);

// Single-producer (audio callback) / single-consumer (main loop) ring of
// interleaved ch0,ch1 samples. Power-of-two size, indices free-run.
constexpr uint32_t kRingSize = 8192;
static int16_t ring_buf[kRingSize];
static volatile uint32_t ring_w = 0;
static volatile uint32_t ring_r = 0;

static float acc0 = 0.0f, acc1 = 0.0f;
static int   acc_n = 0;

static inline int16_t FloatToS16(float x)
{
    if (x > 1.0f) x = 1.0f;
    if (x < -1.0f) x = -1.0f;
    return static_cast<int16_t>(x * 32767.0f);
}

void AudioCallback(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        // Boxcar average of kDecimation samples = cheap anti-alias + decimate
        acc0 += in[0][i];
        acc1 += in[1][i];
        acc_n++;

        if (acc_n == kDecimation)
        {
            int16_t s0 = FloatToS16(acc0 * kGain / kDecimation);
            int16_t s1 = FloatToS16(acc1 * kGain / kDecimation);
            acc0 = acc1 = 0.0f;
            acc_n = 0;

            uint32_t w = ring_w;
            if (w + 2 - ring_r <= kRingSize)
            {
                ring_buf[w % kRingSize]       = s0;
                ring_buf[(w + 1) % kRingSize] = s1;
                ring_w = w + 2;
            }
            // else: host not draining — drop newest samples
        }
    }
}

int main(void)
{
    hw.Init();
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_96KHZ);
    hw.SetAudioBlockSize(kBlockSize);

    // Non-blocking so the board streams whether or not a host is attached.
    // Also gives us the "reboot" command for button-free flashing.
    SerialLibrary serial(hw);
    serial.Init(false);

    hw.StartAudio(AudioCallback);

    static uint8_t frame[kFrameSize];
    frame[0] = 0xDA;
    frame[1] = 0x7A;
    frame[3] = 0x00;
    uint8_t seq = 0;

    while (1)
    {
        // Handle "reboot" (DFU for make flash) from the host
        serial.Poll();

        if (ring_w - ring_r >= kSamplesPerFrame * 2)
        {
            frame[2] = seq++;
            int16_t* payload = reinterpret_cast<int16_t*>(&frame[kHeaderSize]);
            uint32_t r = ring_r;
            for (size_t i = 0; i < kSamplesPerFrame * 2; i++)
            {
                payload[i] = ring_buf[(r + i) % kRingSize];
            }
            ring_r = r + kSamplesPerFrame * 2;

            // Retry briefly if the USB TX buffer is busy; drop otherwise
            for (int attempt = 0; attempt < 100; attempt++)
            {
                if (hw.usb_handle.TransmitInternal(frame, kFrameSize)
                    == UsbHandle::Result::OK)
                {
                    break;
                }
            }
        }
    }
}
