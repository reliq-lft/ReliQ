#!/bin/bash
# Wrapper for nvcc that properly separates compiler and linker flags

echo "NVCC_WRAPPER CALLED WITH: $@" >> /tmp/nvcc_wrapper.log

# Check if this is a linking step (no -c flag means linking)
IS_LINKING=false
HAS_COMPILE_ONLY=false

for arg in "$@"; do
    if [[ "$arg" == "-c" ]]; then
        HAS_COMPILE_ONLY=true
        break
    fi
done

if [[ "$HAS_COMPILE_ONLY" == "false" ]]; then
    IS_LINKING=true
fi

if [ "$IS_LINKING" = true ]; then
    echo "LINKING STEP" >> /tmp/nvcc_wrapper.log
    # Linking step - extract flags from -Xcompiler and process them
    LINK_FLAGS=()
    for arg in "$@"; do
        if [[ "$arg" == -Xcompiler=* ]]; then
            # Extract flags from -Xcompiler
            flags="${arg#-Xcompiler=}"
            # Remove quotes
            flags="${flags#\"}"
            flags="${flags%\"}"
            # Split by space and process
            IFS=' ' read -ra ITEMS <<< "$flags"
            for item in "${ITEMS[@]}"; do
                if [[ "$item" == -L* ]] || [[ "$item" == -l* ]]; then
                    # Library paths and flags - pass directly (not in -Xcompiler)
                    LINK_FLAGS+=("$item")
                elif [[ "$item" == -Wl,-rpath,* ]]; then
                    # Convert -Wl,-rpath,PATH to -Xlinker -rpath -Xlinker PATH
                    path="${item#-Wl,-rpath,}"
                    LINK_FLAGS+=("-Xlinker" "-rpath" "-Xlinker" "$path")
                elif [[ "$item" == -pthread ]]; then
                    # -pthread needs to go through compiler, not linker
                    LINK_FLAGS+=("-Xcompiler=-pthread")
                # else: drop other flags that were in -Xcompiler (they're not needed for linking)
                fi
            done
        else
            # Not in -Xcompiler, pass through
            LINK_FLAGS+=("$arg")
        fi
    done
    
    echo "FINAL LINK COMMAND: /usr/bin/nvcc ${LINK_FLAGS[@]}" >> /tmp/nvcc_wrapper.log
    # Execute nvcc with separated flags
    exec /usr/bin/nvcc "${LINK_FLAGS[@]}"
else
    echo "COMPILE STEP - PASS THROUGH" >> /tmp/nvcc_wrapper.log
    # Compilation step - pass through
    exec /usr/bin/nvcc "$@"
fi
