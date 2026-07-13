#include "serial_library.h"

// Static instance pointer for callback
SerialLibrary* SerialLibrary::instance_ = nullptr;

SerialLibrary::SerialLibrary(DaisySeed& hw) : hw_(hw), command_buffer_("") {
    instance_ = this;
}

SerialLibrary::~SerialLibrary() {
    if (instance_ == this) {
        instance_ = nullptr;
    }
}

void SerialLibrary::Init() {
    // Start the log and wait for connection
    hw_.StartLog(true);
    
    // Set USB callback
    hw_.usb_handle.SetReceiveCallback(UsbCallback, UsbHandle::UsbPeriph::FS_INTERNAL);
}

bool SerialLibrary::HasData() {
    return !msg_fifo_.IsEmpty();
}

int SerialLibrary::GetChar() {
    if (msg_fifo_.IsEmpty()) {
        return 0;
    }
    return msg_fifo_.PopFront();
}

bool SerialLibrary::CheckCommand(const char* command) {
    // Process any new data and add to command buffer
    while (HasData()) {
        int ch = GetChar();
        if (ch >= 32 && ch <= 126) { // Printable ASCII characters
            command_buffer_ += static_cast<char>(ch);
        } else if (ch == '\n' || ch == '\r') {
            // End of line found, check if command matches
            std::string cmd = command_buffer_;
            command_buffer_.clear();
            
            // Remove any trailing whitespace
            while (!cmd.empty() && (cmd.back() == ' ' || cmd.back() == '\t')) {
                cmd.pop_back();
            }
            
            return (cmd == command);
        }
    }
    
    // Also check if the current buffer matches the command (without newline)
    if (!command_buffer_.empty()) {
        std::string cmd = command_buffer_;
        
        // Remove any trailing whitespace
        while (!cmd.empty() && (cmd.back() == ' ' || cmd.back() == '\t')) {
            cmd.pop_back();
        }
        
        if (cmd == command) {
            command_buffer_.clear(); // Clear for next command
            return true;
        } else if (cmd.length() >= strlen(command)) {
            // If buffer is longer than expected command, clear it
            command_buffer_.clear();
        }
    }
    
    return false;
}

// Static callback function
void SerialLibrary::UsbCallback(uint8_t* buff, uint32_t* length) {
    if (instance_ && buff && length) {
        // Push all received bytes to the FIFO
        for (uint32_t i = 0; i < *length; i++) {
            instance_->msg_fifo_.PushBack(buff[i]);
        }
    }
} 