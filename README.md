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
# Installation
Provide clear instructions on how to install the necessary software, dependencies, or setup required for this project.

# Preparation
Outline the steps needed to prepare the environment, configure the settings, or gather any necessary resources before running the application.

# Testing
Detail how to test the application, including specific commands, inputs, and expected outputs for verifying its functionality.

# Autoprocess
Explain the automation process, including how it works, what it automates, and how to trigger it effectively.
