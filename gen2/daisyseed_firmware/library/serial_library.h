#pragma once

#include "daisy_seed.h"
#include <string>

using namespace daisy;

class SerialLibrary {
public:
    SerialLibrary(DaisySeed& hw);
    ~SerialLibrary();
    
    // Initialize serial communication
    void Init();
    
    // Check if there's data available
    bool HasData();
    
    // Get a single character (returns 0 if no data)
    int GetChar();
    
    // Check if a specific command was received
    bool CheckCommand(const char* command);

private:
    DaisySeed& hw_;
    FIFO<uint8_t, 1024> msg_fifo_;
    std::string command_buffer_;
    
    // Static callback function for USB reception
    static void UsbCallback(uint8_t* buff, uint32_t* length);
    static SerialLibrary* instance_;
}; 