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

void SerialLibrary::Init(bool wait_for_connection) {
    // Start the log; only block for a host when requested
    hw_.StartLog(wait_for_connection);

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

            // Built-in: drop into the STM DFU bootloader so the host can
            // reflash without touching the BOOT/RESET buttons
            if (cmd == "reboot") {
                System::ResetToBootloader();
            }

            if (command && cmd == command) {
                return true;
            }
            // Non-matching line: keep scanning any remaining input
        }
    }

    // Also check if the current buffer matches the command (without newline)
    if (command && !command_buffer_.empty()) {
        std::string cmd = command_buffer_;

        // Remove any trailing whitespace
        while (!cmd.empty() && (cmd.back() == ' ' || cmd.back() == '\t')) {
            cmd.pop_back();
        }

        if (cmd == command) {
            command_buffer_.clear(); // Clear for next command
            return true;
        } else if (cmd == "reboot") {
            command_buffer_.clear();
            System::ResetToBootloader();
        } else if (cmd.length() >= strlen(command)) {
            // If buffer is longer than expected command, clear it
            command_buffer_.clear();
        }
    }

    return false;
}

void SerialLibrary::Poll() {
    CheckCommand(nullptr);
}

bool SerialLibrary::GetLine(std::string& out) {
    while (HasData()) {
        int ch = GetChar();
        if (ch >= 32 && ch <= 126) { // Printable ASCII characters
            command_buffer_ += static_cast<char>(ch);
        } else if (ch == '\n' || ch == '\r') {
            std::string cmd = command_buffer_;
            command_buffer_.clear();

            // Trim leading/trailing whitespace
            while (!cmd.empty() && (cmd.back() == ' ' || cmd.back() == '\t')) {
                cmd.pop_back();
            }
            while (!cmd.empty() && (cmd.front() == ' ' || cmd.front() == '\t')) {
                cmd.erase(cmd.begin());
            }
            if (cmd.empty()) {
                continue;
            }

            // Built-in: drop into the STM DFU bootloader so the host can
            // reflash without touching the BOOT/RESET buttons
            if (cmd == "reboot") {
                System::ResetToBootloader();
            }

            out = cmd;
            return true;
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
