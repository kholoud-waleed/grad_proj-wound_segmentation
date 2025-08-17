# 🩹 Real-Time Wound Segmentation on OAK-D Lite

This repository contains the complete pipeline of my graduation project, where I developed a **computer vision system for wound analysis**.  
The system integrates **image preprocessing, automated polygon annotations, YOLOv8-based segmentation, and feature extraction** of wound regions, with final deployment on the **Luxonis OAK-D Lite camera** for **real-time inference**.

---

## 📂 Repository Structure

- **`image_preprocessing/`**  
  Scripts and utilities for cleaning, resizing, and normalizing wound images to ensure high-quality input for training and evaluation.

- **`polygon_auto_annot/`**  
  Automatic polygon-based annotation using **Segment Anything Model (SAM)** to speed up dataset labeling.

- **`image_segmentation/`**  
  YOLOv8n segmentation pipeline:  
  - Training on the prepared dataset after preprocessing and labelling
  - Evaluation of model by viewing its metrics and confusion matrix
  - Evaluation on test set by using model to make predictions  
  - Inference results for wound detection and segmentation  

- **`feature_extraction/`**  
  Extraction of clinically relevant shape features (area, perimeter, eccentricity, etc.) from binary wound masks to enable further medical analysis and motion planning for stitching paths generation.

---

## 🚀 Project Workflow

1. **Image Preprocessing**  
   - Resizing, Padding, Greyscale conversion, Normalisation, Filtering, CLAHE and augmentation for dataset preparation.

2. **Polygon Annotation (SAM)**  
   - Automated polygon-wise masks generated for wound regions.  
   - Reduced manual labeling effort via labelme tool.

3. **Segmentation with YOLOv8**  
   - Trained a YOLOv8n segmentation model on the dataset.  
   - Achieved robust segmentation performance on the test set.  

4. **Feature Extraction**  
   - Extracted wound descriptors from binary masks.  
   - Useful for objective wound assessment.

5. **Deployment on OAK-D Lite**  
   - Model is optimized and ready for **real-time wound segmentation** using the DepthAI pipeline on the OAK-D Lite camera.

---

## 🛠️ Tech Stack

- **Languages**: Python  
- **Frameworks & Libraries**:  
  - [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
  - [Segment Anything Model (SAM)](https://docs.ultralytics.com/models/sam/)  
  - OpenCV, NumPy, Pandas, Matplotlib  
- **Hardware**: Luxonis **OAK-D Lite** camera (https://docs.luxonis.com/software/ros/depthai-ros/) 

---

## 📊 Results

- **Accurate wound segmentation** with YOLOv8n  (95.2% mAP@50 for the wound class on the validation set, and a 93.9% recall / 85% F1-score on the test set, confirming its reliability)
- **Automated dataset annotation** using SAM  
- **Feature descriptors** extracted for further clinical use  
- **Deployment-ready model** running on OAK-D Lite for real-time applications  

---

## 👨‍💻 Author

**Kholoud Waleed Ali**  
Mechatronics & Robotics Engineer | AI/ML & Computer Vision Enthusiast  
