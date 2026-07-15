#include "daisy_seed.h"
#include "library/serial_library.h"
#include <cstdlib>
#include <string>

using namespace daisy;

// Streams this board's two codec inputs (hydrophones) to the host as raw PCM
// over USB CDC. Sample rate and bit depth are set at runtime by the host:
//   "rate 96000|48000|32000|24000|16000"   (integer decimations of 96 kHz)
//   "bits 16|24"
//   "reboot"                               (DFU for make flash)
// Boots at 96 kHz / 24-bit. Board-agnostic: flash it on master (hydrophones
// 0/1) or slave (2/3). Host side: research/whisper_ivc on the Orin.
//
// Wire format, little-endian:
//   [0xDA][0x7A][seq u8][fmt u8][32 x (ch0, ch1)]
//   fmt: bit7 = 24-bit samples, bits0-3 = decimation from 96 kHz.
//        0x00 = legacy firmware (16 kHz / 16-bit, decimation 6).
//   Samples are s16, or s24 packed in 3 bytes when the 24-bit flag is set.

////////////////////////////// Configuration (WE CAN CHANGE) /////////////////
constexpr size_t kBlockSize = 48;        // audio callback block size
const float      kGain      = 100.0f;    // same input gain as level firmware

////////////////////////////// Internal (DO NOT CHANGE) //////////////////////
DaisySeed hw;

// One frame per 64-sample block = one FFT block of the detection firmwares,
// so host-side TDOA frames align 1:1 with detection blocks. Flagged by fmt
// bit6 (legacy 32-sample frames have it clear).
constexpr size_t kSamplesPerFrame = 64;  // per channel, per USB frame
constexpr size_t kHeaderSize      = 4;
constexpr size_t kMaxFrameSize = kHeaderSize + kSamplesPerFrame * 2 * 3;
// Batch frames per USB write: one small write at a time tops out around
// 264 kB/s (per-transfer overhead); 96 kHz/16-bit needs ~400 kB/s. The CDC
// TX buffer is 2048 bytes, so batch as many frames as fit.
constexpr size_t kTxBufSize = 2048;

// Host-controlled runtime config
static volatile int g_decimation   = 1;  // 96 kHz
static volatile int g_sample_bytes = 3;  // 24-bit

// Single-producer (audio callback) / single-consumer (main loop) ring of
// interleaved ch0,ch1 samples, kept as float so a bit-depth change never
// costs precision. Power-of-two size, indices free-run.
// ~340 ms at 96 kHz stereo: must absorb host-side GIL/scheduling stalls
// (the kernel tty buffer only covers ~110 ms at this data rate).
constexpr uint32_t kRingSize = 65536;
static float DSY_SDRAM_BSS ring_buf[kRingSize];  // too big for internal RAM
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

static inline int32_t FloatToS24(float x)
{
    if (x > 1.0f) x = 1.0f;
    if (x < -1.0f) x = -1.0f;
    return static_cast<int32_t>(x * 8388607.0f);
}

// 0 = unsupported rate
static int DecimationForRate(int hz)
{
    switch (hz)
    {
        case 96000: return 1;
        case 48000: return 2;
        case 32000: return 3;
        case 24000: return 4;
        case 16000: return 6;
    }
    return 0;
}

void AudioCallback(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size)
{
    int dec = g_decimation;
    for (size_t i = 0; i < size; i++)
    {
        // Boxcar average of dec samples = cheap anti-alias + decimate
        acc0 += in[0][i];
        acc1 += in[1][i];
        acc_n++;

        if (acc_n >= dec)
        {
            float s0 = acc0 * kGain / acc_n;
            float s1 = acc1 * kGain / acc_n;
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

    static uint8_t txbuf[kTxBufSize];
    uint8_t seq = 0;

    std::string line;
    while (1)
    {
        // Host commands ("reboot" is handled inside GetLine)
        while (serial.GetLine(line))
        {
            if (line.rfind("rate ", 0) == 0)
            {
                int dec = DecimationForRate(atoi(line.c_str() + 5));
                if (dec)
                {
                    g_decimation = dec;
                    ring_r = ring_w;  // drop samples taken at the old rate
                    hw.PrintLine("cfg: rate=%d", 96000 / dec);
                }
                else
                {
                    hw.PrintLine("cfg: bad rate (96000/48000/32000/24000/16000)");
                }
            }
            else if (line == "bits 16" || line == "bits 24")
            {
                g_sample_bytes = (line == "bits 24") ? 3 : 2;
                hw.PrintLine("cfg: bits=%d", g_sample_bytes * 8);
            }
        }

        // A batch that couldn't be sent stays pending and is retried on the
        // next loop pass (commands above stay serviced): discarding it on a
        // short timeout turns every host-side stall into a sequence gap.
        static size_t   pending_bytes = 0;
        static uint32_t pending_t0    = 0;
        if (pending_bytes)
        {
            if (hw.usb_handle.TransmitInternal(txbuf, pending_bytes)
                == UsbHandle::Result::OK)
            {
                pending_bytes = 0;
            }
            else if (System::GetUs() - pending_t0 > 500000)
            {
                pending_bytes = 0;  // host gone for 0.5 s: give up on batch
            }
            continue;
        }

        int    bytes      = g_sample_bytes;
        size_t frame_size = kHeaderSize + kSamplesPerFrame * 2 * bytes;

        // Pack as many complete frames as are ready (up to the buffer cap)
        // into one buffer, then push them in a single USB write.
        size_t   max_frames = kTxBufSize / frame_size;
        size_t   n_frames   = 0;
        uint8_t* p          = txbuf;
        while (n_frames < max_frames
               && ring_w - ring_r >= kSamplesPerFrame * 2)
        {
            p[0] = 0xDA;
            p[1] = 0x7A;
            p[2] = seq++;
            p[3] = static_cast<uint8_t>((bytes == 3 ? 0x80 : 0x00)
                                        | 0x40 /* 64-sample frames */
                                        | (g_decimation & 0x0F));
            p += kHeaderSize;
            uint32_t r = ring_r;
            for (size_t i = 0; i < kSamplesPerFrame * 2; i++)
            {
                float s = ring_buf[(r + i) % kRingSize];
                if (bytes == 2)
                {
                    int16_t v = FloatToS16(s);
                    *p++ = static_cast<uint8_t>(v);
                    *p++ = static_cast<uint8_t>(v >> 8);
                }
                else
                {
                    int32_t v = FloatToS24(s);
                    *p++ = static_cast<uint8_t>(v);
                    *p++ = static_cast<uint8_t>(v >> 8);
                    *p++ = static_cast<uint8_t>(v >> 16);
                }
            }
            ring_r = r + kSamplesPerFrame * 2;
            n_frames++;
        }

        if (n_frames)
        {
            if (hw.usb_handle.TransmitInternal(txbuf, n_frames * frame_size)
                != UsbHandle::Result::OK)
            {
                pending_bytes = n_frames * frame_size;
                pending_t0    = System::GetUs();
            }
        }
    }
}
