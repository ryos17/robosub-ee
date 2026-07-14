# Program Selection
CURRENT_PROGRAM = master_level

# Project Name
TARGET = $(CURRENT_PROGRAM)

# Sources
CPP_SOURCES = $(CURRENT_PROGRAM).cpp library/fft_library.cpp library/serial_library.cpp

# Library Locations
LIBDAISY_DIR = ../../libDaisy/
DAISYSP_DIR = ../../DaisySP/

# Core location, and generic Makefile.
SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core
include $(SYSTEM_FILES_DIR)/Makefile

# ---------------------------------------------------------------------------
# Button-free flashing.
#
#   make flash                          # build + flash (single Daisy on USB)
#   make flash CURRENT_PROGRAM=slave    # choose which program to build/flash
#   make flash DAISY_PORT=/dev/serial/by-id/usb-Electrosmith_...  # if 2+ boards
#
# Sends "reboot" over the Daisy's USB serial port; firmware built from this
# tree calls System::ResetToBootloader() (see library/serial_library.cpp),
# which lands the MCU in the STM DFU bootloader, then dfu-util programs it.
# A board running firmware WITHOUT the reboot command (or a crashed board)
# needs the buttons once: hold BOOT, tap RESET, release, then `make program-dfu`.
# ---------------------------------------------------------------------------
DAISY_PORTS := $(wildcard /dev/serial/by-id/usb-Electrosmith_Daisy_Seed*)
DAISY_PORT ?= $(DAISY_PORTS)

.PHONY: flash
flash: all
	@if lsusb -d 0483:df11 >/dev/null 2>&1; then \
	    echo "Daisy already in DFU mode, programming..."; \
	else \
	    ports="$(DAISY_PORT)"; n=$$(echo $$ports | wc -w); \
	    if [ $$n -eq 0 ]; then \
	        echo "No Daisy serial port and no DFU device found."; \
	        echo "Hold BOOT, tap RESET, release, then re-run 'make flash'."; \
	        exit 1; \
	    fi; \
	    if [ $$n -gt 1 ]; then \
	        echo "Multiple Daisy boards found:"; printf '  %s\n' $$ports; \
	        echo "Re-run with DAISY_PORT=<one of the above>"; \
	        exit 1; \
	    fi; \
	    echo "Rebooting $$ports into DFU..."; \
	    stty -F $$ports 115200 raw -echo || true; \
	    echo reboot > $$ports || true; \
	    sleep 2; \
	fi
	$(MAKE) program-dfu
