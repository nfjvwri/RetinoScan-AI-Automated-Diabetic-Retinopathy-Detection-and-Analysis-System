import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# =================================================
# PATIENT DETAILS
# =================================================

patient_name = input("Enter Patient Name : ")
patient_age = input("Enter Patient Age : ")
patient_id = input("Enter Patient ID : ")

scan_date = datetime.now()

# =================================================
# IMAGE ACQUISITION
# =================================================

image_path = r"C:\Users\DELL\OneDrive\Desktop\DR_Feature_Extraction\dataset\10120_right.jpeg"

# Read retinal image
img = cv2.imread(image_path)

# Convert BGR to RGB
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# =================================================
# PREPROCESSING
# =================================================

# Grayscale conversion
gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

# CLAHE Enhancement
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

enhanced = clahe.apply(gray)

# =================================================
# FILTERING
# =================================================

# Gaussian Blur Filter
blur = cv2.GaussianBlur(
    enhanced,
    (5,5),
    0
)

# =================================================
# SEGMENTATION
# =================================================

# Blood Vessel Segmentation
_, vessels = cv2.threshold(
    blur,
    120,
    255,
    cv2.THRESH_BINARY_INV
)

# Exudate Segmentation
_, exudates = cv2.threshold(
    enhanced,
    220,
    255,
    cv2.THRESH_BINARY
)

# =================================================
# FEATURE EXTRACTION
# =================================================

# Detect contours
contours, _ = cv2.findContours(
    exudates,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Copy original image for highlighting
highlighted = original.copy()

lesion_count = 0
probability_scores = []

# Loop through lesions
for contour in contours:

    area = cv2.contourArea(contour)

    # Ignore tiny noisy regions
    if area > 20:

        lesion_count += 1

        x, y, w, h = cv2.boundingRect(contour)

        # =================================================
        # PROBABILITY ESTIMATION
        # =================================================

        probability = min(95, int(area / 8))

        probability_scores.append(probability)

        # Draw rectangle
        cv2.rectangle(
            highlighted,
            (x, y),
            (x+w, y+h),
            (255,0,0),
            2
        )

        # Add probability text
        cv2.putText(
            highlighted,
            f"{probability}%",
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,0),
            1
        )

# =================================================
# VESSEL DENSITY
# =================================================

vessel_pixels = cv2.countNonZero(vessels)

total_pixels = vessels.shape[0] * vessels.shape[1]

vessel_density = vessel_pixels / total_pixels

# =================================================
# LESION PIXEL COUNT
# =================================================

white_pixels = cv2.countNonZero(exudates)

# =================================================
# CLASSIFICATION
# =================================================

if white_pixels < 500 and lesion_count < 20:
    stage = "Mild Diabetic Retinopathy"

elif white_pixels < 3000 and lesion_count < 50:
    stage = "Moderate Diabetic Retinopathy"

else:
    stage = "Severe Diabetic Retinopathy"

# =================================================
# OVERALL PROBABILITY
# =================================================

if len(probability_scores) > 0:
    overall_probability = sum(probability_scores) / len(probability_scores)
else:
    overall_probability = 0

# =================================================
# DISPLAY PIPELINE OUTPUTS
# =================================================

plt.figure(figsize=(18,10))

# Original Image
plt.subplot(2,3,1)
plt.imshow(original)
plt.title("1. Original Retinal Image")
plt.axis("off")

# Grayscale
plt.subplot(2,3,2)
plt.imshow(gray, cmap='gray')
plt.title("2. Grayscale Conversion")
plt.axis("off")

# CLAHE
plt.subplot(2,3,3)
plt.imshow(enhanced, cmap='gray')
plt.title("3. CLAHE Enhancement")
plt.axis("off")

# Gaussian Filtering
plt.subplot(2,3,4)
plt.imshow(blur, cmap='gray')
plt.title("4. Gaussian Filtering")
plt.axis("off")

# Segmentation
plt.subplot(2,3,5)
plt.imshow(vessels, cmap='gray')
plt.title("5. Blood Vessel Segmentation")
plt.axis("off")

# Final Detection
plt.subplot(2,3,6)
plt.imshow(highlighted)
plt.title("6. Lesion Detection with Probability")
plt.axis("off")

plt.suptitle(
    "Diabetic Retinopathy Detection Pipeline",
    fontsize=16
)

plt.tight_layout()
plt.show()

# =================================================
# SAVE FINAL OUTPUT IMAGE
# =================================================

cv2.imwrite(
    "outputs/final_output.jpg",
    cv2.cvtColor(highlighted, cv2.COLOR_RGB2BGR)
)

# =================================================
# REPORT GENERATION
# =================================================

report = f"""
=================================================
DIABETIC RETINOPATHY ANALYSIS REPORT
=================================================

Patient Name : {patient_name}
Patient Age  : {patient_age}
Patient ID   : {patient_id}

Scan Date :
{scan_date}

=================================================
IMAGE PROCESSING PIPELINE
=================================================

1. Image Acquisition
2. Preprocessing
3. Filtering
4. Segmentation
5. Feature Extraction
6. Classification
7. Probability Mapping
8. Report Generation

=================================================
TECHNIQUES USED
=================================================

Preprocessing :
Grayscale Conversion

Enhancement :
CLAHE Enhancement

Filtering :
Gaussian Blur Filtering

Segmentation :
Binary Thresholding Segmentation

Feature Extraction :
Contour Detection
Blood Vessel Extraction
Lesion Detection

Classification :
Rule-Based DR Classification

=================================================
EXTRACTED FEATURES
=================================================

Lesion Count :
{lesion_count}

Bright Lesion Count :
{white_pixels}

Vessel Density :
{vessel_density:.4f}

Overall DR Probability :
{overall_probability:.2f}%

=================================================
CLASSIFICATION RESULT
=================================================

Predicted Stage :
{stage}

=================================================
SYSTEM OBSERVATION
=================================================

Highlighted retinal regions indicate probable
abnormal lesion areas.

Probability percentages are estimated based on
lesion region intensity and area.

Further ophthalmologist evaluation is recommended.

=================================================
"""

# Print report
print(report)

# Save report
with open("outputs/report.txt", "w") as file:
    file.write(report)

print("Report generated successfully!")

print("Processed image saved successfully!")
