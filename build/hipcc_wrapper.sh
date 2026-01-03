#!/bin/bash
# hipcc wrapper script to fix library linking with Global Arrays
# Similar to nvcc_wrapper.sh but for AMD HIP/ROCm
#
# Problem: Nim passes all --passL flags wrapped in -Xcompiler="..."
# hipcc treats everything in -Xcompiler as compiler flags, not linker flags
# Solution: Extract library flags and pass them directly to hipcc during linking

LOG_FILE="/tmp/hipcc_wrapper.log"
echo "=== hipcc_wrapper called at $(date) ===" >> "$LOG_FILE"
echo "Args: $@" >> "$LOG_FILE"

# Find real hipcc
REAL_HIPCC=$(which -a hipcc | grep -v "$(readlink -f "$0")" | head -1)
echo "Using hipcc: $REAL_HIPCC" >> "$LOG_FILE"

# Check if this is a linking step (no -c flag present)
IS_LINKING=true
for arg in "$@"; do
    if [ "$arg" = "-c" ]; then
        IS_LINKING=false
        break
    fi
done

if [ "$IS_LINKING" = true ]; then
    echo "Linking step detected" >> "$LOG_FILE"
    
    # Process arguments to extract library flags from -Xcompiler
    NEW_ARGS=()
    for arg in "$@"; do
        if [[ "$arg" =~ ^-Xcompiler=(.+)$ ]]; then
            # Extract content from -Xcompiler wrapper
            COMPILER_FLAGS="${BASH_REMATCH[1]}"
            echo "Processing -Xcompiler=$COMPILER_FLAGS" >> "$LOG_FILE"
            
            # Split on spaces and process each flag
            IFS=' ' read -ra FLAGS <<< "$COMPILER_FLAGS"
            for flag in "${FLAGS[@]}"; do
                if [[ "$flag" =~ ^-L.+ ]] || [[ "$flag" =~ ^-l.+ ]]; then
                    # Library flags: pass directly to hipcc
                    echo "  Extracting library flag: $flag" >> "$LOG_FILE"
                    NEW_ARGS+=("$flag")
                elif [[ "$flag" =~ ^-Wl,-rpath,(.+)$ ]]; then
                    # Convert -Wl,-rpath,PATH to -Xlinker -rpath -Xlinker PATH
                    RPATH="${BASH_REMATCH[1]}"
                    echo "  Converting rpath: $RPATH" >> "$LOG_FILE"
                    NEW_ARGS+=("-Xlinker" "-rpath" "-Xlinker" "$RPATH")
                elif [ "$flag" = "-pthread" ]; then
                    # pthread needs to be wrapped for gcc compatibility
                    echo "  Wrapping pthread flag" >> "$LOG_FILE"
                    NEW_ARGS+=("-Xcompiler=-pthread")
                else
                    # Other compiler flags: keep wrapped
                    echo "  Keeping wrapped: $flag" >> "$LOG_FILE"
                    NEW_ARGS+=("-Xcompiler=$flag")
                fi
            done
        else
            # Not an -Xcompiler arg, pass through unchanged
            NEW_ARGS+=("$arg")
        fi
    done
    
    echo "Executing: $REAL_HIPCC ${NEW_ARGS[@]}" >> "$LOG_FILE"
    exec "$REAL_HIPCC" "${NEW_ARGS[@]}"
else
    # Compilation step: pass through unchanged
    echo "Compilation step - passing through" >> "$LOG_FILE"
    exec "$REAL_HIPCC" "$@"
fi
