"""
GPU detection and ROCm environment utilities.
"""

import os
import logging
import subprocess
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def is_rocm_available() -> bool:
    """Check if ROCm is available on the system."""
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    return os.path.exists(rocm_path)


def get_gpu_info() -> Dict[str, any]:
    """Get information about available GPUs.
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        "available": False,
        "count": 0,
        "name": "Unknown",
        "memory_total": 0,
        "memory_free": 0,
        "is_rocm": False,
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            info["available"] = True
            info["count"] = torch.cuda.device_count()
            info["name"] = torch.cuda.get_device_name(0)
            
            # Get memory info
            memory_info = torch.cuda.mem_get_info(0)
            info["memory_free"] = memory_info[0]
            info["memory_total"] = memory_info[1]
            
            # Check if using ROCm
            info["is_rocm"] = is_rocm_available()
            
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
    
    return info


def setup_rocm_env():
    """Setup optimal ROCm environment variables."""
    if not is_rocm_available():
        return
    
    # Default ROCm optimizations
    defaults = {
        "GPU_MAX_HW_QUEUES": "8",
        "HSA_ENABLE_SDMA": "0",
        "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
    }
    
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.debug(f"Set {key}={value}")


def get_rocm_version() -> Optional[str]:
    """Get installed ROCm version.
    
    Returns:
        ROCm version string or None if not found
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Try alternative method
    version_file = "/opt/rocm/.info/version"
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            return f.read().strip()
    
    return None


def print_gpu_diagnostics():
    """Print detailed GPU diagnostics for troubleshooting."""
    info = get_gpu_info()
    
    print("=" * 50)
    print("GPU Diagnostics")
    print("=" * 50)
    print(f"GPU Available: {info['available']}")
    print(f"GPU Count: {info['count']}")
    print(f"GPU Name: {info['name']}")
    print(f"Memory Total: {info['memory_total'] / 1e9:.1f} GB")
    print(f"Memory Free: {info['memory_free'] / 1e9:.1f} GB")
    print(f"Using ROCm: {info['is_rocm']}")
    
    if info['is_rocm']:
        rocm_version = get_rocm_version()
        print(f"ROCm Version: {rocm_version}")
    
    print("=" * 50)
