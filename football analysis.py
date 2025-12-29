import cv2
import time
import math
import numpy as np
import pandas as pd
import base64
import requests
from pathlib import Path
from tqdm import tqdm
from types import SimpleNamespace
import torch
import random

from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.engine.results import Results, Boxes 

# ==============================================================================
# Real-time Football Analytics Pipeline
# ==============================================================================

# --- Monkey-patch Results class if necessary ---
# This helps if BoTSORT is trying to access results.conf directly
# and the __getattr__ isn't working as expected for your ultralytics version.
if not hasattr(Results, 'conf'):
    print("Attempting to monkey-patch Results class to add 'conf' property.")
    def get_results_conf(self_results):
        if self_results.boxes is not None and hasattr(self_results.boxes, 'conf'):
            return self_results.boxes.conf
        
        device = 'cpu'
        if self_results.boxes is not None and hasattr(self_results.boxes, 'data') and self_results.boxes.data is not None:
            device = self_results.boxes.data.device
        return torch.empty(0, device=device)

    Results.conf = property(get_results_conf)




VIDEO_PATH = Path(r"video.mp4") 
OUTPUT_DIR = Path(r"football_analytics_output") 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_BBOXED_VID = True
METERS_PER_PIXEL = 0.05  
ROBOFLOW_API_KEY = "l1esJLa9JcwyH15lohJc" 
ROBOFLOW_MODEL = "football-players-detection-3zvbc"
ROBOFLOW_VERSION = 12
ROBOFLOW_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}?api_key={ROBOFLOW_API_KEY}"


PLAYER_CLASS_ID = 0      


# --- Colors ---
GOALKEEPER_COLOR_BGR = (0, 0, 255)     # Red for goalkeeper
FOOTBALL_COLOR_BGR = (0, 255, 255)     # Yellow for football
DEFAULT_PLAYER_COLOR = (0, 255, 0)     # Default Green
INFO_TEXT_COLOR = (255, 255, 255)      # White text for info boxes
DOTTED_LINE_COLOR = (192, 192, 192)    # Light gray for distance lines

player_track_colors = {}

# -------------------------------
# Helper Functions
# -------------------------------
def get_dynamic_player_color(track_id, class_id=PLAYER_CLASS_ID): 
    global player_track_colors
   
    if track_id not in player_track_colors:
       
        while True:
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            
            player_track_colors[track_id] = color
            break
    return player_track_colors.get(track_id, DEFAULT_PLAYER_COLOR)

def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    # Draw lines
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    # Draw arcs for corners
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_dotted_line_circles(img, pt1, pt2, color, num_dots=20): 
    for k in range(num_dots + 1):
        if k == 0 : continue 
        alpha = k / num_dots
        x = int(pt1[0] * (1 - alpha) + pt2[0] * alpha)
        y = int(pt1[1] * (1 - alpha) + pt2[1] * alpha)
        cv2.circle(img, (x, y), 1, color, -1)


def detect_with_roboflow(frame_img):
    
    _, img_encoded = cv2.imencode(".jpg", frame_img)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")

    payload = {"image": img_base64}

    
    try:
        response = requests.post(ROBOFLOW_URL, json=payload, timeout=5) 
        response.raise_for_status() 
        predictions = response.json().get("predictions", [])
    except requests.exceptions.RequestException as e:
        print(f"Roboflow API request failed: {e}")
        predictions = []
    except ValueError as e: 
        print(f"Failed to decode Roboflow API response: {e}")
        predictions = []

    detections = []
    for p in predictions:
       
        if "x" in p and "y" in p and "width" in p and "height" in p and "confidence" in p:
            if p["confidence"] < 0.3: 
                continue
            detections.append([
                p["x"] - p["width"] / 2,  # x1
                p["y"] - p["height"] / 2,  # y1
                p["x"] + p["width"] / 2,  # x2
                p["y"] + p["height"] / 2,  # y2
                p["confidence"],           
                PLAYER_CLASS_ID            
                
            ])
    return detections


cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if SAVE_BBOXED_VID:
    output_video_path = str(OUTPUT_DIR / f"{VIDEO_PATH.stem}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
    print(f"Output video will be saved to: {output_video_path}")


args = SimpleNamespace(
    track_high_thresh=0.3,   
    track_low_thresh=0.1,     
    new_track_thresh=0.4,     
    track_buffer=int(fps * 2), 
    match_thresh=0.8,         
    
    gmc_method='none',          
    
    cmc_method='none',
    
    fast_reid_config=Path('fast_reid/configs/MOT17/sbs_S50.yml'), 
    fast_reid_weights=Path('osnet_x0_25_msmt17.pt'), 
    proximity_thresh=0.5,
    appearance_thresh=0.25,
    with_reid=False,
    
)
tracker = BOTSORT(args=args, frame_rate=fps)

prev_centers_for_speed = {} 
records = []
frame_idx = 0


total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
with tqdm(total=total_frames) as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            pbar.n = total_frames 
            pbar.refresh()
            break
        
        original_frame_for_results = frame.copy()

       
        detections = detect_with_roboflow(frame)
        det_array = np.array(detections) if detections else np.empty((0, 6))

        # 2. Prepare Results object for BoTSORT
        
        results = Results(orig_img=original_frame_for_results, path=str(VIDEO_PATH), names={PLAYER_CLASS_ID: 'player'}) # Add other classes to names if model supports
        
        if det_array.shape[0] > 0:
            
            det_tensor = torch.tensor(det_array, dtype=torch.float32).cpu() 
            results.boxes = Boxes(det_tensor, orig_shape=original_frame_for_results.shape[:2])
        else:
            results.boxes = Boxes(torch.empty((0, 6), dtype=torch.float32).cpu(), orig_shape=original_frame_for_results.shape[:2])

        # 3. Track
        
        tracks_data = tracker.update(results, frame) 

        current_frame_player_centers = {} 
        if tracks_data is not None and tracks_data.shape[0] > 0:
            for t_data in tracks_data:
                x1, y1, x2, y2 = map(int, t_data[:4])
                track_id = int(t_data[4])
                
                class_id = PLAYER_CLASS_ID 
                
                

                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

               
                speed_mps = 0.0
                if track_id in prev_centers_for_speed:
                    prev_cx, prev_cy, prev_frame_idx = prev_centers_for_speed[track_id]
                    time_interval_frames = frame_idx - prev_frame_idx
                    if time_interval_frames > 0:
                        time_interval_sec = time_interval_frames / fps
                        dist_pixels = math.hypot(cx - prev_cx, cy - prev_cy)
                        dist_meters = dist_pixels * METERS_PER_PIXEL
                        speed_mps = dist_meters / time_interval_sec
                
                
                if track_id not in prev_centers_for_speed or \
                   math.hypot(cx - prev_centers_for_speed[track_id][0], cy - prev_centers_for_speed[track_id][1]) > 2: 
                    prev_centers_for_speed[track_id] = (cx, cy, frame_idx)


                # Store record
                records.append({
                    "frame": frame_idx, "id": track_id, "cls": "player",
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "center_x_px": cx, "center_y_px": cy,
                    "center_x_m": cx * METERS_PER_PIXEL, "center_y_m": cy * METERS_PER_PIXEL,
                    "speed_mps": speed_mps
                })

                # --- Drawing ---
                player_color = get_dynamic_player_color(track_id, class_id)

                
                info_box_h = 30
                info_box_w = max(80, x2 - x1) 
                info_y1 = y2 + 5
                info_y2 = info_y1 + info_box_h
                info_x1 = int(cx - info_box_w / 2)
                info_x2 = int(cx + info_box_w / 2)
                
                
                info_x1 = max(0, info_x1)
                info_y1 = max(0, info_y1)
                info_x2 = min(W, info_x2)
                info_y2 = min(H, info_y2)

                if info_y2 > info_y1 and info_x2 > info_x1 :
                    draw_rounded_rectangle(frame, (info_x1, info_y1), (info_x2, info_y2), player_color, 2, radius=8)
                    label = f"ID:{track_id} {speed_mps:.1f}m/s"
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    text_x = info_x1 + (info_box_w - text_size[0]) // 2 # Center text
                    text_y = info_y1 + (info_box_h + text_size[1]) // 2
                    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, INFO_TEXT_COLOR, 1, cv2.LINE_AA)

                
                foot_y = y2
                current_frame_player_centers[track_id] = (int(cx), int(foot_y), class_id)


        
        player_ids_in_frame = list(current_frame_player_centers.keys())
        for i in range(len(player_ids_in_frame)):
            for j in range(i + 1, len(player_ids_in_frame)):
                id1 = player_ids_in_frame[i]
                id2 = player_ids_in_frame[j]

                
                pt1_data = current_frame_player_centers[id1]
                pt2_data = current_frame_player_centers[id2]
                pt1_center = (pt1_data[0], pt1_data[1])
                pt2_center = (pt2_data[0], pt2_data[1])

                pixel_dist = math.hypot(pt1_center[0] - pt2_center[0], pt1_center[1] - pt2_center[1])
                real_dist_m = pixel_dist * METERS_PER_PIXEL

                draw_dotted_line_circles(frame, pt1_center, pt2_center, DOTTED_LINE_COLOR, num_dots=15)

                mid_point = ((pt1_center[0] + pt2_center[0]) // 2, (pt1_center[1] + pt2_center[1]) // 2)
                cv2.putText(frame, f"{real_dist_m:.1f}m", (mid_point[0] - 15, mid_point[1] - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)


        if SAVE_BBOXED_VID:
            writer.write(frame)

       
        cv2.imshow("Football Analytics", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
        pbar.update(1)

cap.release()
if SAVE_BBOXED_VID:
    writer.release()
cv2.destroyAllWindows()

# -------------------------------
# Save CSV
# -------------------------------
if records:
    df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / f"{VIDEO_PATH.stem}_tracks.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Tracking data saved to: {csv_path}")
else:
    print("\nNo tracking records to save.")

print(f"✅ Pipeline complete! Annotated video (if saved) is in {OUTPUT_DIR}")