# Troubleshooting CUDA / bitsandbytes Initialization Errors

## Problem

You may encounter errors when running Python scripts that utilize the `bitsandbytes` library for model quantization (e.g., loading QLoRA models with `transformers`), even if `nvidia-smi` shows your GPU and CUDA driver are working, and `torch.cuda.is_available()` returns `True`.

## Symptoms

Common error messages include:

*   `IndexError: list index out of range` originating from `bitsandbytes/cuda_specs.py` (often in the `get_compute_capabilities()[-1]` line).
*   `Could not load bitsandbytes native library`
*   `CUDA Setup failed despite CUDA being available.`
*   `RuntimeError: Failed to import transformers.integrations.bitsandbytes ...` (with the `IndexError` or potentially `AssertionError: Invalid device id` listed as the cause).
*   Errors related to `torch/_dynamo/device_interface.py` or Triton (`IndexError: list index out of range`) occurring *after* the initial `bitsandbytes` errors.
*   Before applying fixes, `python -m bitsandbytes` might show warnings like `CUDA runtime files not found in any environmental path.`
*   Before applying fixes, attempting to load the specific CUDA runtime library via `ctypes` in Python might fail with `cannot open shared object file: No such file or directory`.

## Root Causes

This issue typically stems from one or both of the following:

1.  **Missing CUDA Runtime Symbolic Link:** Your CUDA toolkit installation might have the core runtime library (e.g., `libcudart.so.12.1.55`), but `bitsandbytes` or its dependencies might be explicitly looking for a slightly different filename (e.g., `libcudart.so.12.1`). Standard dynamic linking might not resolve this if the exact name is requested.
2.  **Initialization Order Dependency:** `bitsandbytes` needs to query CUDA device properties during its setup. If it's initialized (often triggered by `transformers` when loading a quantized model) *before* the PyTorch CUDA context is fully prepared and stable, it can fail to get the necessary device information, leading to the `IndexError`.

## Solutions

Apply these solutions in order:

### 1. Verify Environment and Create Missing Symbolic Link

Ensure the CUDA library path is correctly set and create the specific symbolic link `bitsandbytes` might be looking for.

a.  **Check `LD_LIBRARY_PATH`:** Make sure the environment variable includes the path to your CUDA installation's `lib64` directory.
    ```bash
    echo $LD_LIBRARY_PATH
    # Example output should contain: /usr/local/cuda-12.1/lib64
    ```
    If it's missing or incorrect, export it correctly before running your script:
    ```bash
    # Adjust cuda-12.1 to your actual CUDA version
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
    ```

b.  **Identify Exact Runtime Library:** Find the specific versioned `libcudart.so` file.
    ```bash
    # Adjust cuda-12.1 to your actual CUDA version
    ls -l /usr/local/cuda-12.1/lib64/libcudart.*
    ```
    Note the full filename, for example, `libcudart.so.12.1.55`.

c.  **Create the Symbolic Link:** Create a link using the base version number that `bitsandbytes` / PyTorch expects (e.g., `libcudart.so.12.1` if your PyTorch is built for CUDA 12.1 and the file from step (b) was `libcudart.so.12.1.55`).
    ```bash
    # Adjust paths and filenames based on your setup from step (b)
    # Use sudo if required
    sudo ln -s /usr/local/cuda-12.1/lib64/libcudart.so.12.1.55 /usr/local/cuda-12.1/lib64/libcudart.so.12.1
    ```

d.  **Verify Link:** Run the `ls -l` command from step (b) again to confirm the new link exists.

### 2. Explicitly Initialize PyTorch CUDA in Script

Modify your Python script to initialize PyTorch and its CUDA context *before* importing libraries like `transformers` or your application code that uses `bitsandbytes`.

Add the following lines at the **very top** of your main Python script:

```python
# --- Explicit PyTorch CUDA Initialization ---
import torch
try:
    if torch.cuda.is_available():
        # This line forces PyTorch to initialize CUDA, query devices,
        # and set up the context for the default device (GPU 0).
        _ = torch.tensor([0.0], device='cuda:0')
        print(f"INFO: Explicitly initialized PyTorch CUDA on device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: PyTorch reports CUDA is not available.")
except Exception as e:
    print(f"ERROR: Failed during explicit PyTorch CUDA initialization: {e}")
# --- End Initialization ---

# Now import your other libraries and run your code
import transformers
# import your_project_modules...

# ... rest of your script ...