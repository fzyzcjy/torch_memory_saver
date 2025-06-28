#!/bin/bash

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get Python version and platform info
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PLATFORM=$(python -c "import platform; print(platform.machine())")

# Build the path to torch_memory_saver_cpp.abi3.so
SO_PATH="$SCRIPT_DIR/../build/lib.linux-${PLATFORM}-cpython-${PYTHON_VERSION//./}/torch_memory_saver_cpp.abi3.so"

# Check if the .so file exists
if [ ! -f "$SO_PATH" ]; then
    echo "Error: $SO_PATH not found!"
    echo "Python version: $PYTHON_VERSION"
    echo "Platform: $PLATFORM"
    echo "Please make sure torch_memory_saver is built correctly for your Python version."
    exit 1
fi

echo "Using LD_PRELOAD=$SO_PATH"
echo "Running rl_example.py..."

# Set LD_PRELOAD environment variable and run the Python script
LD_PRELOAD="$SO_PATH" python "$SCRIPT_DIR/rl_example.py" 