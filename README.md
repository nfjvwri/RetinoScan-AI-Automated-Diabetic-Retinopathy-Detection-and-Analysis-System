# RetinoScan-AI: Automated Diabetic Retinopathy Detection and Analysis System

## Overview

RetinoScan-AI is a Computer Vision and Medical Image Processing based system developed to assist in the early detection and analysis of Diabetic Retinopathy (DR) using retinal fundus images.

The system performs preprocessing, blood vessel segmentation, lesion detection, probability mapping, stage classification, and automated report generation to help identify retinal abnormalities associated with diabetic retinopathy.

---

## Features

* Retinal image preprocessing
* CLAHE contrast enhancement
* Gaussian filtering
* Blood vessel segmentation
* Exudate and lesion detection
* Lesion highlighting with probability estimation
* Vessel density analysis
* Rule-based DR stage classification
* Automated patient report generation
* Visualization of complete image processing pipeline

---

## Technologies Used

* Python
* OpenCV
* NumPy
* Matplotlib
* Computer Vision
* Image Processing Techniques

---

## Processing Pipeline

1. Image Acquisition
2. Grayscale Conversion
3. CLAHE Enhancement
4. Gaussian Filtering
5. Blood Vessel Segmentation
6. Exudate Segmentation
7. Lesion Detection
8. Feature Extraction
9. Probability Mapping
10. DR Stage Classification
11. Automated Report Generation

---

## Techniques Implemented

### Preprocessing

* Grayscale Conversion
* CLAHE (Contrast Limited Adaptive Histogram Equalization)

### Filtering

* Gaussian Blur Filtering

### Segmentation

* Binary Thresholding
* Blood Vessel Extraction
* Exudate Segmentation

### Feature Extraction

* Contour Detection
* Lesion Area Extraction
* Vessel Density Calculation

### Classification

Rule-based diabetic retinopathy stage prediction:

* Mild Diabetic Retinopathy
* Moderate Diabetic Retinopathy
* Severe Diabetic Retinopathy

---

## Output Features

The system generates:

* Highlighted lesion regions
* Probability percentages for detected lesions
* Vessel density estimation
* Predicted DR severity stage
* Automated diagnostic report

---

## Example Outputs

* Original retinal image
* Enhanced retinal image
* Segmented blood vessels
* Lesion detection visualization
* Final DR classification result

---

## How to Run

### Install Dependencies

```bash
pip install opencv-python matplotlib numpy
```

### Run the Project

```bash
python dr_detection.py
```

---

## Project Structure

```text
RetinoScan-AI/
│
├── dataset/
├── outputs/
├── dr_detection.py
├── README.md
```

---

## Future Improvements

* Deep Learning based classification
* CNN/ResNet integration
* Real-time retinal image analysis
* Web application deployment
* Clinical dataset validation
* Multi-stage DR grading using AI

---

## Applications

* Early diabetic retinopathy screening
* Medical image analysis
* Ophthalmology support systems
* Healthcare AI applications
* Biomedical image processing research

---

## Author

Atreyee Bhattacharyya
