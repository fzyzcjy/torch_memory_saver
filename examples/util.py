import subprocess


def print_gpu_memory_gb(gpu_id=0, message=None):
    """Print GPU memory usage with optional message"""
    if isinstance(gpu_id, str):
        # If first parameter is a string, treat it as message and use default GPU 0
        message = gpu_id
        gpu_id = 0
    
    cmd = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id={gpu_id}"
    result = subprocess.check_output(cmd, shell=True, text=True).strip()
    
    if message:
        print(f"GPU {gpu_id} memory: {int(result) / 1024:.2f} GB ({message})")
    else:
        print(f"GPU {gpu_id} memory: {int(result) / 1024:.2f} GB")