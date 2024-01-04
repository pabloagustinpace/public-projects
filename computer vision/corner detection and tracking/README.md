# Corner Detection and Tracking Project

## Description
This project focuses on implementing a Harris corner detector and tracking corners over time using patch templates and SIFT descriptors. It's tailored for those interested in computer vision and image processing.

## Requirements
- Python 3.x
- Libraries: Numpy, Scipy, OpenCV
- Dataset: Sample images from KITTI Visual Odometry (e.g., first 200 images of sequence 00)

## Installation
- Clone the repository
- Install the required Python packages:
  ```bash
pip install numpy scipy opencv-python

## Repository Structure
- `README.md`: Overview and instructions
- `corner_tracking_patch.mp4`: Video demonstration of corner tracking
- `images/`: Sample images from the KITTI dataset for processing
- `output_images/`: Processed images with detected and tracked corners
- `main.ipynb`: Jupyter Notebook with implementation and examples
- `utils.py`: Utility functions for corner detection and tracking

## Usage
- Navigate to the repository directory.
- Run `main.ipynb` in Jupyter Notebook to view the implementation and examples.
- Utilize `utils.py` for custom corner detection and tracking functionalities.

## Acknowledgments
Special thanks to Prof. Renato Martins (Universitê de Bourgogne) for providing the guidelines and resources for this project.