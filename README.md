
<h1 align="center">Imaging Particle Analysis System</h1>

# Introduction
The Imaging Particle Analysis System is an advanced tool designed for precise and efficient analysis of particle images. This system combines cutting-edge image processing techniques with robust segmentation and data analysis capabilities to deliver actionable insights about particle characteristics such as size and shape. Its modular architecture ensures flexibility and scalability, making it suitable for applications in materials science, pharmaceuticals, and industrial quality control.

At the core of the system is the SAM2 (Segment Anything Model 2), which powers the particle segmentation module, enabling highly accurate and adaptive segmentation of particles from raw images. The segmented data is then processed through a series of specialized models for further analysis, including size filtering and data export.

# Core Features
- **Image Processing Module**

    - Allows users to define a cropping zone and layering parameters.

    - Processes folder-based image inputs for seamless batch analysis.

- **Particle Segmentation Module**

  - Utilizes the state-of-the-art SAM2 model to segment particles with high precision.

  - Extracts critical attributes such as:

      - Area (μm²)

      - Perimeter (μm)

      - Diameter (μm)

      - Circularity

- **Size Analysis Module**

   - Provides size filtering capabilities to categorize particles into specific bins.

   - Exports processed data in XML format for integration with other systems.

- **Automated Data Transfer**

  - Segmented images, analyzed data, and related metadata are saved to specified folders.

  - Facilitates seamless data transfer to external systems for further processing.

# Particle Analyzer Process Start Model

## Overview
This system monitors for image files, processes them using image analysis, and manages the workflow between different servers.

## Configuration

### Setting up config.ini

The system uses `config.ini` for configuration. Key settings include:

```ini
[SMBServer]
SMB_SERVER = AT-SERVER        # The GPU server hostname
SMB_SHARE = ImageDataShare    # The network share name on the GPU server

[CPUWorkstation]
CPU_SERVER = CP-88ED3A        # The CPU workstation hostname
CPU_FOLDER = ImageDataShare_17424009  # The folder to monitor on the CPU workstation
```

### Required Settings

1. **GPU Server (Required)**
   - `SMB_SERVER`: Hostname of the GPU server
   - `SMB_SHARE`: Network share name on the GPU server

2. **CPU Workstation (Optional)**
   - `CPU_SERVER`: Hostname of the CPU workstation
   - `CPU_FOLDER`: Folder path to monitor on the CPU workstation

## How It Works

1. The system monitors two locations:
   - CPU workstation: `\\CPU_SERVER\CPU_FOLDER`
   - GPU server: `\\SMB_SERVER\SMB_SHARE`

2. When image files (.bmp) and corresponding metadata files (.json) appear on the CPU workstation, they are:
   - Copied to the GPU server
   - Deleted from the CPU workstation after successful transfer

3. The GPU server continuously monitors its folder for image and JSON file pairs and processes them using the image analysis model.

## Starting the System

```
python startModel.py
```

## Logs and Troubleshooting

Check the logs for any issues. The system logs:
- File transfers between servers
- Analysis operations
- Error conditions

If the CPU workstation is not accessible, the system will retry connecting periodically.

## Requirements

- Windows environment with network access
- Python 3.6+
- Network share access to both CPU and GPU servers
- Proper permissions to read/write on network shares

# Installation
Provide clear instructions on how to install the necessary software, dependencies, or setup required for this project.

# Preparation
Outline the steps needed to prepare the environment, configure the settings, or gather any necessary resources before running the application.

# Testing
Detail how to test the application, including specific commands, inputs, and expected outputs for verifying its functionality.

# Autoprocess
Explain the automation process, including how it works, what it automates, and how to trigger it effectively.
=======

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

