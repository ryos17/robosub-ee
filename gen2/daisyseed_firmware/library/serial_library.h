#pragma once

#include "daisy_seed.h"
#include <string>

using namespace daisy;

class SerialLibrary {
public:
    SerialLibrary(DaisySeed& hw);
    ~SerialLibrary();

    // Initialize serial communication.
    // wait_for_connection: block until a USB host opens the port (StartLog).
    // Pass false on boards that must run standalone (e.g. the slave).
    void Init(bool wait_for_connection = true);

    // Check if there's data available
    bool HasData();

    // Get a single character (returns 0 if no data)
    int GetChar();

    // Check if a specific command was received.
    // Any completed line equal to "reboot" resets the board into the STM DFU
    // bootloader (this is what `make flash` relies on) — every program that
    // calls CheckCommand()/Poll() in its loop is flashable without buttons.
    bool CheckCommand(const char* command);

    // Process pending input for the built-in "reboot" command only.
    // Call this in the main loop of programs that need no other commands.
    void Poll();

private:
    DaisySeed& hw_;
    FIFO<uint8_t, 1024> msg_fifo_;
    std::string command_buffer_;

    // Static callback function for USB reception
    static void UsbCallback(uint8_t* buff, uint32_t* length);
    static SerialLibrary* instance_;
};
