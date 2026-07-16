#!/usr/bin/env bash
#
# Robust Daisy flasher for this two-board setup.
#
#   ./flash_daisy.sh <program> <master|slave>
#
# Examples:
#   ./flash_daisy.sh master_ros   master
#   ./flash_daisy.sh slave        slave
#   ./flash_daisy.sh master_level master
#
# Why this exists (vs plain `make flash`):
#   - Always does a clean build (rm -rf build) so stale objects from a different
#     CURRENT_PROGRAM can't produce a broken binary.
#   - Resolves the target by ROLE, not by a hard-coded serial: the master's USB
#     serial can enumerate garbled on a marginal link (e.g. 376636603433 instead
#     of 376C36533433), so we pick "the Electrosmith that is NOT the slave".
#   - FAILS LOUDLY if the target board isn't on USB (instead of make flash
#     silently writing `reboot` to a bad path and reporting "No DFU capable").
#   - Retries: on a flaky hub the DFU device often enumerates *after* the
#     Makefile's short wait. We send reboot, poll up to 15 s for DFU, program,
#     and retry a few times. Also handles a board already forced into DFU.

set -uo pipefail
cd "$(dirname "$0")"

SLAVE_SERIAL="376C36573433"          # the slave's serial enumerates reliably
MAX_ATTEMPTS=5
DFU_WAIT_S=15

PROG="${1:-}"
ROLE="${2:-}"
if [ -z "$PROG" ] || [ -z "$ROLE" ]; then
    echo "usage: $0 <program> <master|slave>" >&2
    exit 2
fi

list_es() { ls /dev/serial/by-id/ 2>/dev/null | grep -i electrosmith || true; }
dfu_present() { lsusb -d 0483:df11 >/dev/null 2>&1; }

resolve_port() {   # echoes the by-id NAME for the requested role, or nothing
    case "$ROLE" in
        slave)  list_es | grep    "$SLAVE_SERIAL" | head -1 ;;
        master) list_es | grep -v "$SLAVE_SERIAL" | head -1 ;;
        *) echo "ROLE must be 'master' or 'slave'" >&2; exit 2 ;;
    esac
}

echo ">> clean build: $PROG"
rm -rf build
if ! make CURRENT_PROGRAM="$PROG" >/tmp/daisy_build.log 2>&1; then
    echo "!! BUILD FAILED:" >&2; tail -20 /tmp/daisy_build.log >&2; exit 1
fi

NAME="$(resolve_port)"
if [ -z "$NAME" ] && ! dfu_present; then
    echo "!! '$ROLE' board not found and nothing is in DFU." >&2
    echo "   Electrosmith devices currently on USB:" >&2
    list_es | sed 's/^/     /' >&2
    [ "$ROLE" = master ] && echo "   (master = the Electrosmith that is NOT $SLAVE_SERIAL)" >&2
    echo "   -> plug the $ROLE board into a WORKING USB port (avoid the known-bad one)," >&2
    echo "      or force DFU with the buttons: hold BOOT, tap RESET, release." >&2
    exit 1
fi
[ -n "$NAME" ] && echo ">> target: /dev/serial/by-id/$NAME" || echo ">> target: already in DFU"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo ">> attempt $attempt/$MAX_ATTEMPTS"

    if ! dfu_present; then
        NAME="$(resolve_port)"                       # re-resolve (serial/tty can change)
        PORT=""
        [ -n "$NAME" ] && PORT="/dev/serial/by-id/$NAME"
        if [ -n "$PORT" ] && [ -e "$PORT" ]; then
            echo "   sending 'reboot' -> DFU via $PORT"
            stty -F "$PORT" 115200 raw -echo 2>/dev/null || true
            printf 'reboot\n' > "$PORT" 2>/dev/null || true
        else
            echo "   no serial port for $ROLE this round; waiting for DFU anyway"
        fi
        for _ in $(seq 1 "$DFU_WAIT_S"); do dfu_present && break; sleep 1; done
    fi

    if dfu_present; then
        out="$(make program-dfu CURRENT_PROGRAM="$PROG" 2>&1)"
        echo "$out" | grep -E 'File downloaded successfully|No DFU|Error [0-9]' | sed 's/^/   /'
        if echo "$out" | grep -q 'File downloaded successfully'; then
            echo ">> SUCCESS: $PROG flashed to $ROLE"
            sleep 2
            echo ">> now on USB:"; list_es | sed 's/^/     /'
            exit 0
        fi
    else
        echo "   DFU device never appeared (flaky link / bad port?)"
    fi
    sleep 2
done

echo "!! FAILED after $MAX_ATTEMPTS attempts. Likely a marginal USB port/cable/connector." >&2
echo "   Try a different USB port, a different cable, or BOOT+RESET then re-run." >&2
exit 1
