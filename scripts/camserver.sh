#!/bin/zsh
# Start a RealSense frame server as root, with the HID (IMU) shim that stops librealsense 2.56
# from segfaulting on macOS 26 when a D455 is plugged in.  Usage:
#   scripts/camserver.sh 8765                      # wrist D405 (first device)
#   scripts/camserver.sh 8766 408222301818         # boom D455 by serial
set -e
cd "$(dirname "$0")/.."
PORT=${1:-8765}; SERIAL=${2:-}
SHIM=$PWD/scripts/nohid/nohid.dylib
if [ ! -f "$SHIM" ] || [ scripts/nohid/nohid.c -nt "$SHIM" ]; then
  clang -dynamiclib -framework IOKit -framework CoreFoundation -o "$SHIM" scripts/nohid/nohid.c
  codesign -s - -f "$SHIM" 2>/dev/null
fi
ARGS=(--port "$PORT"); [ -n "$SERIAL" ] && ARGS+=(--serial "$SERIAL")
# sudo strips DYLD_* from its own environment, so hand the variable to env(1) instead
exec sudo env DYLD_INSERT_LIBRARIES="$SHIM" .venv/bin/python -m robojev.perception.camserver "${ARGS[@]}"
