# Mosaicing Project

## Introduction
This project focuses on feature matching and mosaicing images, a key concept in computer vision. Using techniques such as keypoint detection, feature matching, and homography estimation, we create image mosaics that stitch together multiple images based on shared features.

## Repository Structure
- `mosaicing.ipynb`: Jupyter Notebook containing the main implementation and demonstration of the mosaicing process.
- `utils.py`: Python script containing all the utility functions used in the project.
- `images/`: Directory containing the images used for creating mosaics.
- `requirements.txt`: List of Python dependencies required to set up the virtual environment.

## Setting Up
1. Clone the repository.
2. Install the dependencies:
pip install -r requirements.txt
3. Launch the Jupyter Notebook:
jupyter notebook mosaicing.ipynb


## Project Overview
The project is divided into several key steps:
1. **Feature Detection**: Using algorithms like FAST and SIFT to detect keypoints in images.
2. **Feature Matching**: Matching keypoints between different images using descriptors.
3. **Homography Estimation**: Using RANSAC to robustly estimate the homography between different image views.
4. **Image Mosaicing**: Warping and stitching images together based on the estimated homographies.

## Usage
Navigate through the `mosaicing.ipynb` notebook to see the detailed implementation and results of each step. The `utils.py` file contains helper functions used throughout the project.

## Results
The final section of the notebook presents the created mosaics, demonstrating the effectiveness of feature matching and homography estimation in creating image mosaics.

## Dependencies
- OpenCV
- Numpy
- Matplotlib
- (other dependencies listed in `requirements.txt`)

## License
[MIT License](LICENSE.md)

## Acknowledgments
Special thanks to Prof. Renato Martins (Universitê de Bourgogne) for providing the guidelines and resources for this project.
