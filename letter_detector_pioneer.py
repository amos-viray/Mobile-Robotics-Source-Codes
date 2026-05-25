#!/usr/bin/env python3

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import DetrImageProcessor, DetrForObjectDetection

# ROS 2 Core Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

# --- CONFIGURATIONS ---
MODEL_DIR = "/home/cameron-waddingham/Documents/paper_subset/detr_model"
WEIGHTS_PATH = "/home/cameron-waddingham/Documents/greek_cnn.pth"

CONFIDENCE_THRESHOLD = 0.9      # DETR bounding box activation threshold
CNN_CONFIDENCE_THRESHOLD = 0.99 # Minimum confidence required for CNN prediction
W = H = 96                      # Input resolution footprint for GreekCNN
IOU_TRACK_THRESHOLD = 0.4       # Overlap ratio required to match boxes across time

LETTERS = {
    0: "alpha", 1: "beta", 2: "gamma", 3: "delta",
    4: "eta", 5: "lambda", 6: "mu", 7: "rho",
    8: "tau", 9: "psi"
}

# --- MODEL DEFINITIONS ---

class GreekCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(GreekCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 12 * 12) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- PREPROCESSING HELPERS ---

def crop_to_white_paper(pil_img, brightness_threshold=200):
    open_cv_image = np.array(pil_img.convert('L'))
    _, thresh = cv2.threshold(open_cv_image, brightness_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return pil_img  

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    if w < 10 or h < 10:
        return pil_img
        
    return pil_img.crop((x, y, x + w, y + h))

def pad_to_square(img):
    w, h = img.size
    max_side = max(w, h)
    delta_w = max_side - w
    delta_h = max_side - h
    padding = (delta_w//2, delta_h//2, delta_w - delta_w//2, delta_h - delta_h//2)
    return ImageOps.expand(img, padding, fill=255)

inference_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Lambda(pad_to_square),
    transforms.Resize((W, H)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0
    return interArea / unionArea

# --- ROS 2 NODE ARCHITECTURE ---

class GreekTrackingNode(Node):
    def __init__(self):
        super().__init__('greek_tracking_node')
        
        # Initialize communication pipelines
        self.bridge = CvBridge()
        
        # Publishers
        self.image_pub = self.create_publisher(Image, '~/annotated_image', 10)
        self.data_pub = self.create_publisher(String, '~/tracking_data', 10)
        
        # Subscriber (Change 'camera/image_raw' to match your actual camera topic if needed)
        self.image_sub = self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)
        
        # Core Model Initialization
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.get_logger().info(f"Initializing tracking architecture onto target: {self.device}")
        
        try:
            self.processor = DetrImageProcessor.from_pretrained(MODEL_DIR)
            self.detr_model = DetrForObjectDetection.from_pretrained(MODEL_DIR).to(self.device).eval()
            
            self.cnn_model = GreekCNN(num_classes=len(LETTERS))
            self.cnn_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device))
            self.cnn_model.to(self.device).eval()
            self.get_logger().info("Neural network engines locked and loaded.")
        except Exception as e:
            self.get_logger().error(f"Failed to compile models: {str(e)}")
            raise e

        # Persistent Tracking State across asynchronous subscriber callbacks
        self.active_tracks = {}
        self.next_track_id = 1
        
        try:
            self.font = ImageFont.load_default()
        except IOError:
            self.font = None

    def image_callback(self, msg):
        try:
            # Convert incoming ROS 2 image array data down into openCV matrices
            cv_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"cv_bridge matrix parsing failure: {str(e)}")
            return

        # Prepare structural transforms for model compliance
        rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        width, height = pil_image.size

        # PHASE 1: DETR Object Detection Anchor Generation
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        pixel_mask = inputs["pixel_mask"].to(self.device) if "pixel_mask" in inputs else None

        with torch.no_grad():
            outputs = self.detr_model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        target_sizes = torch.tensor([[height, width]]).to(self.device)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD)[0]

        annotated_pil = pil_image.copy()
        draw = ImageDraw.Draw(annotated_pil)
        num_detections = len(results["scores"])

        current_frame_detections = []
        for score, box in zip(results["scores"], results["boxes"]):
            box_coords = box.tolist()
            xmin = max(0, int(box_coords[0]))
            ymin = max(0, int(box_coords[1]))
            xmax = min(width, int(box_coords[2]))
            ymax = min(height, int(box_coords[3]))
            current_frame_detections.append([xmin, ymin, xmax, ymax])

        # PHASE 2: Dynamic Spatial Temporal Track Asssociations
        updated_tracks = {}
        tracking_summary_log = []

        if num_detections > 0:
            for box in current_frame_detections:
                best_iou = 0
                best_tid = None
                
                for tid, track_info in self.active_tracks.items():
                    iou = compute_iou(box, track_info["box"])
                    if iou > best_iou and iou >= IOU_TRACK_THRESHOLD:
                        best_iou = iou
                        best_tid = tid
                
                # Perform isolated regional classifications via the CNN
                detr_box_img = pil_image.crop((box[0], box[1], box[2], box[3]))
                cropped_box_img = crop_to_white_paper(detr_box_img, brightness_threshold=200)
                processed_tensor = inference_transforms(cropped_box_img)
                
                input_batch = processed_tensor.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    cnn_outputs = self.cnn_model(input_batch)
                    probabilities = F.softmax(cnn_outputs, dim=1).squeeze().cpu().numpy()

                if best_tid is not None:
                    # Maintain existing track properties
                    track_data = self.active_tracks[best_tid]
                    track_data["box"] = box
                    track_data["history_probs"].append(probabilities)
                    track_data["frame_count"] += 1
                    track_data["frames_since_seen"] = 0
                    updated_tracks[best_tid] = track_data
                else:
                    # Map unique keying fields for a new tracker profile
                    updated_tracks[self.next_track_id] = {
                        "box": box,
                        "history_probs": [probabilities],
                        "frame_count": 1,
                        "frames_since_seen": 0
                    }
                    self.next_track_id += 1
            
            # Age out tracks that were not updated in this frame
            for tid, track_info in self.active_tracks.items():
                if tid not in updated_tracks:
                    track_info["frames_since_seen"] += 1
                    if track_info["frames_since_seen"] <= 5: # Grace period of 5 frames
                        updated_tracks[tid] = track_info

            self.active_tracks = updated_tracks

            # PHASE 3: Processing & Visualization
            for tid, track_info in self.active_tracks.items():
                if track_info["frames_since_seen"] > 0:
                    continue  # Don't draw box overlays if it wasn't seen in this current frame
                
                box = track_info["box"]
                mean_probs = np.mean(track_info["history_probs"], axis=0)
                best_class = np.argmax(mean_probs)
                running_conf = mean_probs[best_class]
                
                if running_conf < CNN_CONFIDENCE_THRESHOLD:
                    letter_name = "idk"
                else:
                    letter_name = LETTERS.get(best_class, "Unknown")
                
                # Log metrics for data publication
                tracking_summary_log.append(f"id:{tid},char:{letter_name},conf:{running_conf:.4f},frames:{track_info['frame_count']}")
                
                box_color = "yellow" if letter_name == "idk" else "green"
                draw.rectangle(box, outline=box_color, width=4)
                text_str = f" #{tid} {letter_name.upper()} ({running_conf:.1%}) [{track_info['frame_count']}f]"
                text_box = draw.textbbox((box[0], box[1]), text_str, font=self.font)
                draw.rectangle(text_box, fill=box_color)
                draw.text((box[0], box[1]), text_str, fill="black" if box_color == "yellow" else "white", font=self.font)
        else:
            # Drop tracks if the screen is completely empty across multiple execution spans
            for tid in list(self.active_tracks.keys()):
                self.active_tracks[tid]["frames_since_seen"] += 1
                if self.active_tracks[tid]["frames_since_seen"] > 5:
                    del self.active_tracks[tid]

        # Convert back into a BGR matrix format compatible with standard ROS image messages
        annotated_cv = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
        
        # --- PUBLISH DATA & WRITE TO TERMINAL ---
        
        # 1. Standard Terminal Output Logging
        if tracking_summary_log:
            log_str = " | ".join(tracking_summary_log)
            self.get_logger().info(f"Active Targets: [ {log_str} ]")
        else:
            self.get_logger().info("Scanning... No active assets detected.", throttle_duration_sec=1.0)

        # 2. Publish JSON/CSV String Message to tracking data topic
        data_msg = String()
        data_msg.data = ";".join(tracking_summary_log) if tracking_summary_log else "None"
        self.data_pub.publish(data_msg)

        # 3. Publish Annotated Frame to visualization topic
        try:
            ros_image_msg = self.bridge.cv2_to_imgmsg(annotated_cv, encoding="bgr8")
            ros_image_msg.header = msg.header # Maintain original header and timestamp tracking frames
            self.image_pub.publish(ros_image_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to publish annotated frame mapping: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = GreekTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down tracking node via system call parameters.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()