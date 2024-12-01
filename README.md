
# Repository Setup Guide

> **Note**: This repository supports Python up to **v3.12** (PyTorch).

## How to Set Up

### 0. Sync the Repository
Ensure your local repository is up to date.

---

### 1. Install PyTorch
Install PyTorch based on your CUDA version. Visit the [official PyTorch website](https://pytorch.org/) for more details.

For example, if you are using CUDA 11.8, run:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### 2. Install SAM2
Clone the `SAM2` repository and install it:
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

---

### 3. Install Dependencies
Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

You're now ready to use the repository!
