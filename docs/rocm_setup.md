# ROCm Setup Guide

## Installing ROCm

### Ubuntu/Debian

```bash
# Add AMD repository
sudo apt update
sudo apt install wget gnupg2
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.0/ jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list

# Install ROCm
sudo apt update
sudo apt install rocm-hip-sdk rocm-dev

# Add user to render and video groups
sudo usermod -aG render,video $USER

# Reboot
sudo reboot
```

### Verify Installation

```bash
# Check ROCm version
cat /opt/rocm/.info/version

# Check GPU detection
rocm-smi

# Check HIP
hipconfig --version
```

## Environment Variables

```bash
# Add to ~/.bashrc or ~/.profile
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Performance tuning
export GPU_MAX_HW_QUEUES=8
export HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

## PyTorch with ROCm

```bash
# Install PyTorch with ROCm support
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```
