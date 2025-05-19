# Swim Pose Estimation
Traditionally, swim pose estimation was a challenging task, requiring identification of the swimmer of interest, triangulation of the swimmer, identification of orientation of the swimmer, and intricate analysis of the swimmers joint positions.

From breakthroughs in image processing techniques and computer vision, swim pose estimation now seems within reach. Via the Swim Analysis pipeline, the 4 components of which are listed below, we aim to classify the stroke of a swimmer of interest from individual frames in swimming videos:
1. Preprocessing Pipeline: Create a video processing pipeline to process and transform images frame-by-frame
2. Image Segmentation: Segment the swimmer from the background using Meta’s Segment Anything Model (SAM)
3. Keypoint Generation: Detect and label the swimmer’s joints using a human pose estimation model of your choice (either YOLO or Meta Sapiens)
4. Pose Estimation: Classify the pose of the swimmer

## Setup

### Swim Video Download (Training Data)

1. Download all swim video data [here](https://drive.google.com/drive/u/2/folders/1yPP7xIZ8hxW5XJdqeiwhXfWEORRwvrLl)
2. Create data/swim_videos folder in project
3. Move all swim video .mp4 files to this folder (data/swim_videos)

### Download Meta ViT-H SAM model

1. Download [Meta ViT-H SAM Model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
2. Move Meta ViT-H SAM Model to models/ folder

### Setup Virtual Environment
Set up a python virtual environment with the project's dependencies to run Swim Analysis and Pose Estimation Pipelines.
```bash
python -m venv venv
```

Activate the newly created virtual environment.
```bash
source venv/bin/activate
```

Install project dependencies.
```bash
pip install -r requirements.txt
```

## Swim Analysis Pipeline
Run swim_analysis.py to perform video preprocessing, frame-by-frame transformation, image segmentation, and keypoint generation.

```bash
python swim_analysis.py [--start START] [--stop STOP] [--step STEP]
```

Segmented swim images will be saved in data/segmented.
Keypoint generated images will be saved in data/keypoints. Keypoint coordinate data will be saved to data/keypoints/keypoints.csv

## Pose Estimation Pipeline

### Random Forest Classifier & Support Vector Machine
To perform pose estimation using the Random Forest Classifier and Support Vector Machine, run:
```bash
python stroke_classifier.py
```

### Multi-Layer Perceptron (MLP)
To perform pose estimation using the Multi-Layer Perceptron, run:
```bash
python stroke_classifier_mlp.py
```

### Convolutional Neural Network (CNN)
To perform pose estimation using the CNN, run:
```bash
python stroke_classifier_cnn.py
```

### AlexNet
To perform pose estimation using AlexNet, run:
```bash
python stroke_classifier_alexnet.py
```

### ResNet-50
To perform pose estimation using ResNet-50, run:
```bash
python stroke_classifier_resnet.py
```
