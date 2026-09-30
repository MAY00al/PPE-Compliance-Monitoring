# PPE Compliance Monitoring System

An AI-based computer vision system for detecting Personal Protective Equipment (PPE) and potential PPE violations in construction-site images using a fine-tuned YOLO26n object detection model.

## Project Overview

This project develops an object detection system for identifying PPE-related conditions in workplace and construction-site images.

The system detects four classes:

- `Helmet`
- `NoHelmet`
- `NoVest`
- `Vest`

For an input image, the model produces bounding boxes, class labels, and confidence scores. The final model is also integrated into an interactive Gradio interface for image-based PPE detection and violation summarization.

## Problem Description

Manual monitoring of PPE compliance can be time-consuming, particularly when multiple workers or work areas need to be inspected.

The objective of this project is to develop a computer vision system capable of automatically detecting the presence or absence of safety helmets and safety vests.

The expected output is:

**Input Image → PPE Detection → Bounding Boxes + Class Labels + Confidence Scores → Violation Summary**

## Dataset & Model Used

### Dataset

The project uses the **Personal Protective Equipment (PPE) Dataset — Version 6** from Mendeley Data.

**Dataset:** https://data.mendeley.com/datasets/zkzghjvpn2/6

The dataset contains **2,286 images** with existing YOLO-format object detection annotations.

| Split | Images |
|---|---:|
| Train | 1,829 |
| Validation | 228 |
| Test | 229 |
| **Total** | **2,286** |

The four target classes are:

1. Helmet
2. NoHelmet
3. NoVest
4. Vest

### Model

The project uses **YOLO26n** from Ultralytics.

Pretrained YOLO26n weights were used as the initialization point, followed by fine-tuning on the four-class PPE dataset using transfer learning. No layers were manually frozen during training.

Training configuration:

| Parameter | Value |
|---|---|
| Model | YOLO26n |
| Input Size | 640 × 640 |
| Maximum Epochs | 50 |
| Batch Size | 16 |
| Early Stopping Patience | 10 |
| Best Epoch | 38 |
| Stopped Epoch | 48 |

Preprocessing and training-time augmentation were handled through the Ultralytics YOLO training pipeline.

## Workflow / Architecture

![PPE Compliance Monitoring Workflow](system_workflow.png)

The project workflow includes:

**Dataset Preparation → Preprocessing & Training Augmentation → YOLO26n Fine-Tuning → Final Test Evaluation → PPE Detection → Gradio Interactive Demo**

## Results & Evaluation

The final model was evaluated on an unseen test set containing **229 images and 600 annotated instances**.

### Overall Test Performance

| Metric | Score |
|---|---:|
| Precision | 0.586 |
| Recall | 0.734 |
| mAP@50 | 0.682 |
| mAP@50–95 | 0.487 |

### Per-Class Performance

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Helmet | 0.506 | 0.851 | 0.669 | 0.482 |
| NoHelmet | 0.564 | 0.764 | 0.687 | 0.513 |
| NoVest | 0.483 | 0.534 | 0.535 | 0.360 |
| Vest | 0.791 | 0.788 | 0.836 | 0.594 |

The **Vest** class achieved the strongest performance, while **NoVest** was the most challenging class.

The evaluation also includes:

- Normalized confusion matrix
- Precision–Recall curve
- F1–Confidence curve
- Successful prediction example
- Failure case analysis

Detailed evaluation outputs and visualizations are available in the project notebook.

## Technologies Used

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- Matplotlib
- Gradio
- Google Colab
- GitHub

## How to Run the Project

### 1. Open the Notebook

Open:

`PPE_Compliance_Monitoring.ipynb`

The project was developed and tested using **Google Colab with GPU acceleration**.

### 2. Install the Required Packages

```python
!pip install -q ultralytics
!pip install -q gradio
```

### 3. Prepare the Dataset

Download the PPE dataset from Mendeley Data:

https://data.mendeley.com/datasets/zkzghjvpn2/6

Follow the dataset preparation and configuration steps provided in the notebook.

### 4. Run the Notebook

Execute the notebook cells in order to:

- Validate and prepare the dataset
- Create the train, validation, and test splits
- Fine-tune YOLO26n
- Evaluate the final model
- Analyze successful and failed predictions
- Run the interactive Gradio demonstration

## Future Improvements

Future work could include:

- Expanding and diversifying the PPE dataset
- Improving detection of the `NoVest` class and small or partially occluded PPE objects
- Adding person detection and worker-to-PPE association for individual compliance assessment
- Evaluating the system on video and real-time camera streams
- Exploring model export and optimization for edge or production deployment

## Training Program

This project was completed as part of the **Computer Vision Systems Development** training program.

### SDAIA Academy

[SDAIA Academy GitHub](https://github.com/SDAIAAcademy)
