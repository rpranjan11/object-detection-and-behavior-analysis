# Pre-requisite:

# If you don’t have Homebrew:
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install python@3.11 ffmpeg pkg-config

# Create and activate a fresh venv
python3.11 -m venv ~/venvs/yolo-mps
source ~/venvs/yolo-mps/bin/activate

# Core libs: PyTorch (with MPS), Ultralytics, OpenCV, Supervision (for nice annotations/exports)
pip install --upgrade pip
pip install torch torchvision torchaudio  # macOS wheels include MPS support
pip install ultralytics opencv-python supervision

# Sanity check for MPS:
python - << 'PY'
import torch; print("MPS available:", torch.backends.mps.is_available())
PY

# Quick YOLO inference on video:
python detect.py "/full/path/to_video.mp4"

