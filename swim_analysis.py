import os
import sys
import cv2
import torch
import numpy as np
from torchvision.transforms import v2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from ultralytics import YOLO
import supervision as sv

class SwimAnalyzer:
    def __init__(self, sam_checkpoint_path, sam_type, pose_estimation_model_path, device):
        """ SWIM ANALYZER INITIALIZATION """
        # Swim Analyzer Variables
        self.stroke = ''
        
        # Image Segmentation Variables
        self.sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint_path)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam, 
            points_per_side=32, 
            pred_iou_thresh=0.88, 
            stability_score_thresh=0.95, 
            min_mask_region_area=100
        )
        self.mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

        # Pose Estimation Variables
        self.pose_estimation_model = YOLO(pose_estimation_model_path)
    
    def set_stroke(self, stroke):
        """ Set stroke
        
        Args:
            stroke (str): stroke to set.
        """
        self.stroke = stroke
    
    def write_frame(self, output_folder, idx, frame):
        """ Helper function to save frame
        
        Args:
            output_folder (str): output folder.
            idx (int): frame index.
            frame (np.ndarray): frame to save.
        """
        output_dir = f'{output_folder}{self.stroke}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{idx}.png')
        cv2.imwrite(output_path, frame)
        
    def pose_estimation(self, frame, detections, idx):
        """ POSE ESTIMATION 
        
        Args:
            frame (np.ndarray): frame to estimate pose.
            detections (list): detected segmentations masks.
            idx (int): frame index.
        """
        # Pose Estimation on frame
        results = self.pose_estimation_model(frame.copy())[0]
        # results.show()
        
        # Process results list
        keypoints = results.keypoints.xy[0]
                
        if len(keypoints) > 0:
            # Save pose estimated frame
            pose_estimated_frame = results.plot()
            if detections:
                pose_estimated_frame = self.mask_annotator.annotate(scene=pose_estimated_frame, detections=detections)
            self.write_frame(KEYPOINTS_FOLDER, idx, pose_estimated_frame)

            # Save keypoints to csv file
            for x, y in keypoints:
                with open(KEYPOINTS_CSV_FILE, 'a') as f:
                    f.write(f"{self.stroke},{idx},{x},{y}\n")

        else:
            print("No keypoints found")

    def segmentation(self, frame, idx):
        """ IMAGE SEGMENTATION 
        
        Args:
            frame (np.ndarray): frame to segment.
            idx (int): frame index.
        """
        # Convert frame to RGB for SAM
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Generate automatic masks
        masks = self.mask_generator.generate(rgb_frame)
        detections = sv.Detections.from_sam(masks)        
        
        # Apply mask
        segmented_frame = self.mask_annotator.annotate(scene=frame.copy(), detections=detections)
        
        # Save mask to file
        self.write_frame(SEGMENTED_IMAGES_FOLDER, idx, segmented_frame)

        return detections

    def preprocess_frame(self, frame):
        """ IMAGE PREPROCESSING
        
        Args:
            frame (np.ndarray): frame to transform.
        """
        # Transformation pipeline
        transforms = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transforms(frame)

    def process_frame(self, frame, idx):
        """ PROCESS FRAME FUNCTION
        Algorithm:
        1. Image Preprocessing
        2. Image Segmentation
        3. Pose Estimation

        Args:
            frame (np.ndarray): frame to process.
            idx (int): frame index.
        """
        # 1. Image Preprocessing
        # frame = self.preprocess_frame(frame, idx)
        
        # 2. Image Segmentation
        # detections = None
        detections = self.segmentation(frame, idx)

        # 3. Pose estimation
        self.pose_estimation(frame, detections, idx)
            
    def process_video(self, start = 0, stop = 64, step = 1):
        """ Process video by frame, segmenting and estimating pose for every frame.

        Args:
            start (int, optional): start frame index. Defaults to 0.
            stop (int, optional): stop frame index. Defaults to 64.
            step (int, optional): step size. Defaults to 1.
        """
        input_path = f'{SWIM_VIDEO_FOLDER}{self.stroke}-training.mp4'
        cap = cv2.VideoCapture(input_path)
        
        # Process video
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (idx >= stop):
                break
            
            print(f"idx: {idx}")
            if idx >= start and idx % step == 0:
                # Process frame
                self.process_frame(frame, idx)
            
            idx += 1
        
        # Release resources
        cap.release()

def main(start, stop, step, stroke):
    """ SWIM ANALYSIS MAIN FUNCTION """
    
    # Initialize analyzer
    analyzer = SwimAnalyzer(
        sam_checkpoint_path=MODEL_CHECKPOINT_LARGE,
        sam_type=SAM_TYPE_LARGE,
        pose_estimation_model_path=POSE_ESTIMATION_MODEL_PATH,
        device=device
    )
    
    # Process swim videos
    analyzer.set_stroke(stroke)
    analyzer.process_video(start, stop, step)

if __name__ == "__main__":
    # Image dataset folder paths
    PROCESSED_IMAGES_FOLDER = 'data/processed/'
    SWIM_VIDEO_FOLDER = 'data/Swim Videos/'
    INPUT_IMAGES_FOLDER = 'data/input/'
    SEGMENTED_IMAGES_FOLDER = 'data/segmented/'
    KEYPOINTS_FOLDER = 'data/keypoints/'
    KEYPOINTS_CSV_FILE = 'data/keypoints/keypoints.csv'

    # Segmentation variables
    MODEL_CHECKPOINT_LARGE = 'models/sam_vit_h_4b8939.pth'
    SAM_TYPE_LARGE = 'vit_h'

    # Pose Estimation variables
    POSE_ESTIMATION_MODEL_PATH = 'models/yolo11n-pose.pt'

    # Device variables
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run Swim Analysis main function
    start, stop, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    stroke = sys.argv[4]
    main(start, stop, step, stroke)