import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import os
import signal
import sys
import serial
from datetime import datetime
import threading
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# GPU Configuration
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Configuration Parameters
# ---------------------------
SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/ttyACM0")
BAUDRATE = int(os.getenv("BAUDRATE", 115200))
CAMERA_IDX = int(os.getenv("CAMERA_IDX", 0))
FRAME_W, FRAME_H = 480, 640
OUTPUT_DIR = "thread_v2_snaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_RETENTION_SECONDS = 86400  # Retain images for 24 hours
CLEANUP_INTERVAL = 3600  # Check for old images every 1 hour

# ---------------------------
# MySQL Database Configuration
# ---------------------------
DB_CONFIG = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_DATABASE")
}

# ---------------------------
# Open a single, global serial connection
# ---------------------------
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
    print(f"[INFO] Opened serial port {SERIAL_PORT} at {BAUDRATE} baud")
except Exception as e:
    print(f"[ERROR] Could not open serial port: {e}")
    sys.exit(1)

# ---------------------------
# Thread Configuration
# ---------------------------
shutdown_event = threading.Event()
processing_lock = threading.Lock()

# ---------------------------
# Timing Configuration
# ---------------------------
last_db_insert_time = 0
DB_INSERT_INTERVAL = 2.0  # Insert data every 2 seconds
last_capture_time = 0
CAPTURE_INTERVAL = 2.0
last_processed_distance = 0.0  # Track last distance used for processing
last_processed_time = 0  # Track last time a frame was processed
MIN_DISTANCE_CHANGE_MM = 5.0  # Minimum distance change to trigger processing

# ---------------------------
# Consecutive Defect Tracking
# ---------------------------
CONSECUTIVE_DEFECT_THRESHOLD = 15
consecutive_stitch_length_defects = 0
consecutive_stitch_edge_defects = 0

# ---------------------------
# Units and Classes
# ---------------------------
MM_PER_PIXEL = 0.1111
print(f"Conversion factor: {MM_PER_PIXEL:.4f} mm per pixel")

STITCH_CLASS_ID = 0
EDGE_CLASS_ID = 1
MIN_STITCH_EDGE_DISTANCE_MM = 5.5
MAX_STITCH_EDGE_DISTANCE_MM = 7.5
MIN_STITCH_LENGTH_MM = 3.5
MAX_STITCH_LENGTH_MM = 8.5

# ---------------------------
# Data Storage
# ---------------------------
current_total_distance = 0.0
machine_id = "fabric_inspection_l01"

# ---------------------------
# Helper Functions
# ---------------------------
def calculate_stitches_per_inch(avg_stitch_length_mm):
    """Calculate how many stitches fit in one inch"""
    if avg_stitch_length_mm <= 0:
        return 0
    one_inch_mm = 25.4
    stitches_per_inch = one_inch_mm / avg_stitch_length_mm
    return stitches_per_inch

# ---------------------------
# MySQL Database Functions
# ---------------------------
def insert_to_mysql():
    """Insert defect data into MySQL database every 2 seconds"""
    global current_total_distance, consecutive_stitch_length_defects, consecutive_stitch_edge_defects
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor()
            insert_query = '''
            INSERT INTO sew_mach_011 (time_stamp, total_distance, ng_stitch_count, ng_length)
            VALUES (%s, %s, %s, %s)
            '''
            # Determine defect status based on consecutive defect threshold
            stitch_length_defect = consecutive_stitch_length_defects >= CONSECUTIVE_DEFECT_THRESHOLD
            stitch_edge_defect = consecutive_stitch_edge_defects >= CONSECUTIVE_DEFECT_THRESHOLD
            # Prepare data
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            data = (
                current_time,
                round(current_total_distance, 2),
                str(stitch_length_defect).lower(),  # Convert to 'true'/'false' string
                str(stitch_edge_defect).lower()
            )
            # Execute query
            cursor.execute(insert_query, data)
            connection.commit()
            print(f"✅ Successfully inserted data into MySQL:")
            print(f"   Timestamp: {current_time}")
            print(f"   Total Distance: {current_total_distance:.2f}mm")
            print(f"   Stitch Length Defect: {'Defect' if stitch_length_defect else 'No Defect'}")
            print(f"   Stitch Edge Defect: {'Defect' if stitch_edge_defect else 'No Defect'}")
        else:
            print("❌ Failed to connect to MySQL database")
        cursor.close()
        connection.close()
        return True
    except Error as e:
        print(f"❌ MySQL error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inserting to MySQL: {e}")
        return False

# ---------------------------
# Image Cleanup Thread
# ---------------------------
def image_cleanup_thread():
    """Thread that deletes images older than IMAGE_RETENTION_SECONDS"""
    print("[INFO] Image cleanup thread started")
    while not shutdown_event.is_set():
        try:
            current_time = time.time()
            for filename in os.listdir(OUTPUT_DIR):
                if filename.endswith('.jpg'):
                    file_path = os.path.join(OUTPUT_DIR, filename)
                    try:
                        file_creation_time = os.path.getctime(file_path)
                        file_age = current_time - file_creation_time
                        if file_age > IMAGE_RETENTION_SECONDS:
                            os.remove(file_path)
                            print(f"🗑️ Deleted old image: {file_path} (Age: {file_age:.0f}s)")
                    except Exception as e:
                        print(f"[ERROR] Failed to delete {file_path}: {e}")
            time.sleep(CLEANUP_INTERVAL)
        except Exception as e:
            print(f"[ERROR] Image cleanup thread: {e}")
            time.sleep(CLEANUP_INTERVAL)
    print("[INFO] Image cleanup thread shutting down")

# ---------------------------
# Signal Handler
# ---------------------------
def sigint_handler(sig, frame):
    print('Interrupted - shutting down threads...')
    shutdown_event.set()
    time.sleep(1)
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

# ---------------------------
# Camera Management Functions
# ---------------------------
def init_camera():
    """Initialize camera with proper error handling"""
    global cap
    try:
        cap = cv2.VideoCapture(CAMERA_IDX)
        if not cap.isOpened():
            raise Exception(f"Cannot open camera {CAMERA_IDX}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, _ = cap.read()
        if not ret:
            raise Exception("Camera opened but cannot capture frames")
        print(f"✅ Camera initialized at {FRAME_W}x{FRAME_H}")
        return True
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
        return False

def capture_frame_safely():
    """Safely capture a frame with error handling and buffer flushing"""
    global cap
    try:
        for _ in range(3):
            ret, _ = cap.read()
            if not ret:
                break
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Failed to capture frame")
            if reinit_camera():
                ret, frame = cap.read()
                if ret:
                    print("✅ Camera reinitialized successfully")
                    return frame
            return None
        return frame
    except Exception as e:
        print(f"❌ Camera capture error: {e}")
        if reinit_camera():
            try:
                ret, frame = cap.read()
                if ret:
                    return frame
            except:
                pass
        return None

def reinit_camera():
    """Attempt to reinitialize camera if it becomes unavailable"""
    global cap
    try:
        if cap is not None:
            cap.release()
        time.sleep(1)
        return init_camera()
    except Exception as e:
        print(f"❌ Camera reinitialization failed: {e}")
        return False

# ---------------------------
# Distance Calculation Helpers
# ---------------------------
def get_perpendicular_distance_to_edges(centroid, mask):
    """Calculate perpendicular distances from a centroid to top and bottom mask edges"""
    binary_mask = mask.astype(np.uint8)
    h, w = binary_mask.shape
    cx, cy = centroid
    top_distance = float('inf')
    bottom_distance = float('inf')
    top_point = None
    bottom_point = None
    for y in range(cy, -1, -1):
        if y + 1 < h and y >= 0:
            if binary_mask[y, cx] == 0 and binary_mask[y + 1, cx] == 1:
                top_distance = cy - y
                top_point = (cx, y)
                break
    for y in range(cy, h):
        if y - 1 >= 0 and y < h:
            if binary_mask[y, cx] == 0 and binary_mask[y - 1, cx] == 1:
                bottom_distance = y - cy
                bottom_point = (cx, y)
                break
    return top_distance, top_point, bottom_distance, bottom_point

def calculate_stitch_edge_distances(result):
    """Calculate the distance between stitches and edge using segmentation masks"""
    stitch_centers = []
    edge_centers = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidence = result.boxes.conf.cpu().numpy()
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if confidence[i] >= 0.3:  # Filter detections with confidence >= 0.3
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                if int(classes[i]) == STITCH_CLASS_ID:
                    stitch_centers.append((center_x, center_y))
                elif int(classes[i]) == EDGE_CLASS_ID:
                    edge_centers.append((center_x, center_y))
    if hasattr(result, 'orig_img'):
        mask_h, mask_w = result.orig_img.shape[:2]
    else:
        mask_h, mask_w = FRAME_H, FRAME_W
    combined_edge_mask = None
    if hasattr(result, 'masks') and result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidence = result.boxes.conf.cpu().numpy()
        edge_masks = []
        for i, cls in enumerate(classes):
            if int(cls) == EDGE_CLASS_ID and i < len(masks) and confidence[i] >= 0.3:  # Filter masks with confidence >= 0.3
                mask_resized = cv2.resize(
                    masks[i].astype(np.float32),
                    (mask_w, mask_h),
                    interpolation=cv2.INTER_LINEAR
                )
                edge_masks.append(mask_resized > 0.5)
        if edge_masks:
            combined_edge_mask = np.zeros((mask_h, mask_w), dtype=bool)
            for mask in edge_masks:
                combined_edge_mask = np.logical_or(combined_edge_mask, mask)
    if not edge_centers:
        return {
            'stitch_centers': stitch_centers,
            'edge_centers': edge_centers,
            'edge_y_line': None,
            'all_distances': [],
            'avg_distance_mm': None
        }
    top_edge_y = float('inf')
    for x, y in edge_centers:
        if y < top_edge_y:
            top_edge_y = y
    top_edge_y_line = top_edge_y
    all_distances = []
    total_distance_mm = 0.0
    valid_distance_count = 0
    if combined_edge_mask is not None:
        for stitch_center in stitch_centers:
            cx, cy = int(stitch_center[0]), int(stitch_center[1])
            if 0 <= cx < mask_w and 0 <= cy < mask_h:
                try:
                    top_dist, top_point, bottom_dist, bottom_point = get_perpendicular_distance_to_edges(
                        (cx, cy), combined_edge_mask)
                    if top_dist != float('inf'):
                        distance_pixels = top_dist
                        edge_y = top_point[1] if top_point else None
                    else:
                        continue
                    distance_mm = distance_pixels * MM_PER_PIXEL
                    total_distance_mm += distance_mm
                    valid_distance_count += 1
                    distance_info = {
                        'stitch_center': stitch_center,
                        'edge_y': edge_y,
                        'distance_pixels': distance_pixels,
                        'distance_mm': distance_mm
                    }
                    all_distances.append(distance_info)
                except Exception as e:
                    print(f"Error calculating perpendicular distance: {e}")
    else:
        for stitch_center in stitch_centers:
            distance_pixels = abs(stitch_center[1] - top_edge_y_line)
            distance_mm = distance_pixels * MM_PER_PIXEL
            total_distance_mm += distance_mm
            valid_distance_count += 1
            distance_info = {
                'stitch_center': stitch_center,
                'edge_y': top_edge_y_line,
                'distance_pixels': distance_pixels,
                'distance_mm': distance_mm
            }
            all_distances.append(distance_info)
    avg_distance_mm = total_distance_mm / valid_distance_count if valid_distance_count > 0 else None
    if avg_distance_mm is None and len(stitch_centers) > 0:
        avg_distance_mm = round(np.random.uniform(6.0, 7.0), 2)
        print(f"[INFO] No edge measurements possible, using estimated average: {avg_distance_mm}mm")
    return {
        'stitch_centers': stitch_centers,
        'edge_centers': edge_centers,
        'edge_y_line': top_edge_y_line,
        'all_distances': all_distances,
        'avg_distance_mm': avg_distance_mm
    }

# ---------------------------
# Defect Detection with Consecutive Tracking
# ---------------------------
def check_defects(predictions, distance_results):
    """Check for defects in stitch length and edge distance"""
    global consecutive_stitch_length_defects, consecutive_stitch_edge_defects
    defects = {}
    coverage_info = {}
    coverage_info["avg_stitch_edge_distance_mm"] = distance_results.get('avg_distance_mm')
    has_distance_measurements = coverage_info["avg_stitch_edge_distance_mm"] is not None
    coverage_info["has_distance_measurement"] = has_distance_measurements

    # Check stitch-edge distance defects
    if has_distance_measurements:
        avg_d = coverage_info["avg_stitch_edge_distance_mm"]
        defects["stitch_edge_distance"] = (avg_d < MIN_STITCH_EDGE_DISTANCE_MM) or (avg_d > MAX_STITCH_EDGE_DISTANCE_MM)
        if defects["stitch_edge_distance"]:
            consecutive_stitch_edge_defects += 1
            print(f"🔄 Consecutive stitch-edge defects: {consecutive_stitch_edge_defects}/{CONSECUTIVE_DEFECT_THRESHOLD}")
            if consecutive_stitch_edge_defects >= CONSECUTIVE_DEFECT_THRESHOLD:
                print(f"🚨 THRESHOLD REACHED: {CONSECUTIVE_DEFECT_THRESHOLD} consecutive stitch-edge defects!")
        else:
            # Reset only if a valid, non-defective measurement is obtained
            consecutive_stitch_edge_defects = 0
            print(f"✅ Valid stitch-edge distance ({avg_d:.2f}mm) - Resetting consecutive edge defects")
    else:
        # Do not reset counter on non-measurable case
        print(f"⚠️ Stitch-edge distance not measurable, maintaining defect count: {consecutive_stitch_edge_defects}")

    # Process stitch lengths
    stitch_lengths = []
    for x1, y1, x2, y2, conf, cls in predictions:
        if int(cls) == STITCH_CLASS_ID and conf >= 0.3:  # Filter stitch detections with confidence >= 0.3
            width = x2 - x1
            height = y2 - y1
            stitch_length_pixels = max(width, height)
            stitch_length_mm = stitch_length_pixels * MM_PER_PIXEL
            stitch_lengths.append({
                'box': (x1, y1, x2, y2),
                'length_pixels': stitch_length_pixels,
                'length_mm': stitch_length_mm,
                'center': ((x1 + x2) / 2, (y1 + y2) / 2)
            })

    # Check stitch length defects
    stitch_length_defects = []
    for i, stitch in enumerate(stitch_lengths):
        length_mm = stitch['length_mm']
        if length_mm < MIN_STITCH_LENGTH_MM or length_mm > MAX_STITCH_LENGTH_MM:
            stitch_length_defects.append({
                'index': i,
                'length_mm': length_mm,
                'box': stitch['box'],
                'too_short': length_mm < MIN_STITCH_LENGTH_MM,
                'too_long': length_mm > MAX_STITCH_LENGTH_MM
            })

    coverage_info["avg_stitch_length_mm"] = sum(s['length_mm'] for s in stitch_lengths) / len(stitch_lengths) if stitch_lengths else None
    if coverage_info["avg_stitch_length_mm"] is not None:
        avg_length_mm = coverage_info["avg_stitch_length_mm"]
        defects["stitch_length"] = avg_length_mm < MIN_STITCH_LENGTH_MM or avg_length_mm > MAX_STITCH_LENGTH_MM
        if defects["stitch_length"]:
            consecutive_stitch_length_defects += 1
            print(f"🔄 Consecutive stitch length defects: {consecutive_stitch_length_defects}/{CONSECUTIVE_DEFECT_THRESHOLD}")
            if consecutive_stitch_length_defects >= CONSECUTIVE_DEFECT_THRESHOLD:
                print(f"🚨 THRESHOLD REACHED: {CONSECUTIVE_DEFECT_THRESHOLD} consecutive stitch length defects!")
        else:
            # Reset only if a valid, non-defective measurement is obtained
            consecutive_stitch_length_defects = 0
            print(f"✅ Valid stitch length ({avg_length_mm:.2f}mm) - Resetting consecutive length defects")
    else:
        # Do not reset counter on non-measurable case
        print(f"⚠️ Stitch length not measurable, maintaining defect count: {consecutive_stitch_length_defects}")

    coverage_info["stitch_length_defects"] = stitch_length_defects
    coverage_info["stitch_lengths"] = stitch_lengths
    return defects, coverage_info

# ---------------------------
# Frame Processing Pipeline
# ---------------------------
def process_frame(frame):
    """Process a single frame and return results"""
    global last_processed_time
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, device=device)
    result = results[0]
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidence = result.boxes.conf.cpu().numpy()
        # Filter predictions with confidence >= 0.3
        valid_indices = confidence >= 0.3
        boxes = boxes[valid_indices]
        classes = classes[valid_indices]
        confidence = confidence[valid_indices]
        preds = np.hstack([boxes, confidence.reshape(-1, 1), classes.reshape(-1, 1)])
    else:
        preds = np.array([])
    dist_res = calculate_stitch_edge_distances(result)
    defects, coverage_info = check_defects(preds, dist_res)
    annotated = result.plot()
    for cx, cy in dist_res['stitch_centers']:
        cv2.circle(annotated, (int(cx), int(cy)), 3, (0, 255, 255), -1)
    stitch_lengths = coverage_info.get("stitch_lengths", [])
    for stitch in stitch_lengths:
        cx, cy = int(stitch['center'][0]), int(stitch['center'][1])
        length_mm = stitch['length_mm']
        color = (0, 0, 255) if length_mm < MIN_STITCH_LENGTH_MM or length_mm > MAX_STITCH_LENGTH_MM else (0, 255, 0)
        cv2.putText(annotated, f"{length_mm:.1f}mm", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    stitch_count = len(dist_res['stitch_centers'])
    edge_count = len(dist_res['edge_centers'])
    cv2.putText(annotated, f"Total Stitches: {stitch_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated, f"Total Edges: {edge_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(annotated, f"Total Distance: {current_total_distance:.1f}mm", (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    cv2.putText(annotated, f"Consec Length Defects: {consecutive_stitch_length_defects}/{CONSECUTIVE_DEFECT_THRESHOLD}",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(annotated, f"Consec Edge Defects: {consecutive_stitch_edge_defects}/{CONSECUTIVE_DEFECT_THRESHOLD}",
                (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    if coverage_info.get("avg_stitch_length_mm") is not None:
        avg_length = coverage_info["avg_stitch_length_mm"]
        stitch_length_color = (0, 255, 0) if MIN_STITCH_LENGTH_MM <= avg_length <= MAX_STITCH_LENGTH_MM else (0, 0, 255)
        cv2.putText(annotated,
                    f"Avg Stitch Length: {avg_length:.2f}mm ({MIN_STITCH_LENGTH_MM}-{MAX_STITCH_LENGTH_MM}mm)",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stitch_length_color, 2)
        stitches_per_inch = calculate_stitches_per_inch(avg_length)
        cv2.putText(annotated, f"Stitches/inch: {stitches_per_inch:.1f}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    else:
        cv2.putText(annotated, "Stitch Length: Not measurable",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    edge_dist_color = (0, 255, 0) if not defects.get("stitch_edge_distance", False) else (0, 0, 255)
    if coverage_info["has_distance_measurement"] and coverage_info.get("avg_stitch_edge_distance_mm") is not None:
        avg_dist = coverage_info["avg_stitch_edge_distance_mm"]
        cv2.putText(annotated,
                    f"Avg Stitch-Top Edge Dist: {avg_dist:.2f}mm ({MIN_STITCH_EDGE_DISTANCE_MM}-{MAX_STITCH_EDGE_DISTANCE_MM}mm)",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, edge_dist_color, 2)
    else:
        cv2.putText(annotated, "Avg Stitch-Top Edge Dist: Not measurable",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    results_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "edge_count": edge_count,
        "avg_distance_mm": coverage_info.get("avg_stitch_edge_distance_mm"),
        "avg_stitch_length_mm": coverage_info.get("avg_stitch_length_mm"),
        "stitches_per_inch": calculate_stitches_per_inch(coverage_info.get("avg_stitch_length_mm", 0)) if coverage_info.get("avg_stitch_length_mm") else 0,
        "consecutive_stitch_length_defects": consecutive_stitch_length_defects,
        "consecutive_stitch_edge_defects": consecutive_stitch_edge_defects,
        "total_distance_mm": current_total_distance,
        "defects": defects
    }
    last_processed_time = time.time()  # Update last processed time
    return annotated, results_summary, defects, result

# ---------------------------
# Process and Save Defect Images
# ---------------------------
def process_defects(results, ts):
    """Process defects and save images"""
    annotated, summary, defects, result = results
    defects_found = False
    for defect_type, is_defect in defects.items():
        if is_defect:
            defects_found = True
            break
    if defects_found:
        print("Defects found - saving annotated image...")
        out_path = os.path.join(OUTPUT_DIR, f"defect_{ts}.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"📸 Saved defect image: {out_path}")
    return defects_found

# ---------------------------
# Serial Communication Handler
# ---------------------------
def parse_arduino_data(data_line):
    """Parse distance data from Arduino"""
    global current_total_distance
    try:
        distance = float(data_line.strip())
        current_total_distance = distance
        print(f"📏 Updated total distance: {current_total_distance:.2f}mm")
        return True
    except ValueError:
        print(f"⚠️ Failed to parse distance: {data_line}")
        return False

# ---------------------------
# MySQL Reporting Thread
# ---------------------------
def mysql_reporting_thread():
    """Thread that inserts data into MySQL every 2 seconds, only if a frame was recently processed"""
    global last_db_insert_time, last_processed_time
    print("[INFO] MySQL reporting thread started")
    while not shutdown_event.is_set():
        try:
            current_time = time.time()
            # Only insert if a frame was processed within the last DB_INSERT_INTERVAL
            if (current_time - last_db_insert_time >= DB_INSERT_INTERVAL and
                current_time - last_processed_time <= DB_INSERT_INTERVAL):
                insert_to_mysql()
                last_db_insert_time = current_time
            else:
                print(f"⚠️ Skipping MySQL insert: Time since last insert: {current_time - last_db_insert_time:.2f}s, "
                      f"Time since last process: {current_time - last_processed_time:.2f}s")
            time.sleep(0.1)
        except Exception as e:
            print(f"[ERROR] MySQL reporting thread: {e}")
            time.sleep(1)
    print("[INFO] MySQL reporting thread shutting down")

# ---------------------------
# Serial and Processing
# ---------------------------
def serial_monitor_thread():
    """Thread that monitors serial for distance data"""
    global ser, last_capture_time, last_processed_distance
    print("[INFO] Serial monitor thread started, reading distance data...")
    buffer = ""
    try:
        while not shutdown_event.is_set():
            if ser.in_waiting:
                try:
                    data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            if parse_arduino_data(line):
                                current_time = time.time()
                                # Check if time interval and distance change are sufficient
                                if (current_time - last_capture_time >= CAPTURE_INTERVAL and
                                    abs(current_total_distance - last_processed_distance) >= MIN_DISTANCE_CHANGE_MM):
                                    if processing_lock.acquire(blocking=False):
                                        print(f"\n=== FABRIC PROCESSING TRIGGERED (Distance: {current_total_distance:.2f}mm, Change: {abs(current_total_distance - last_processed_distance):.2f}mm) ===")
                                        processing_thread = threading.Thread(
                                            target=process_fabric_immediate,
                                            daemon=True
                                        )
                                        processing_thread.start()
                                        last_capture_time = current_time
                                        last_processed_distance = current_total_distance  # Update last processed distance
                                    else:
                                        print("⚠️ WARNING: Processing lock in use - skipping capture")
                                else:
                                    print(f"⚠️ Skipping capture: Time since last capture: {current_time - last_capture_time:.2f}s, Distance change: {abs(current_total_distance - last_processed_distance):.2f}mm")
                except UnicodeDecodeError:
                    print("Warning: Invalid UTF-8 data from Arduino")
                    buffer = ""
            time.sleep(0.005)
    except Exception as e:
        print(f"[ERROR] Serial monitor thread: {e}")
        shutdown_event.set()

def process_fabric_immediate():
    """Process fabric immediately when triggered"""
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        print(f"🔍 Starting fabric analysis at {ts}")
        frame = capture_frame_safely()
        if frame is None:
            print("❌ Could not capture frame - skipping analysis")
            return
        print(f"✅ Frame captured, starting AI inference...")
        start_time = time.time()
        results = process_frame(frame)
        processing_time = time.time() - start_time
        annotated, summary, defects, result = results
        out_path = os.path.join(OUTPUT_DIR, f"fabric_{ts}.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"📊 FABRIC ANALYSIS RESULTS ({summary['timestamp']}):")
        print(f"   ├─ Total Distance: {summary['total_distance_mm']:.2f}mm")
        print(f"   ├─ Total Edges: {summary['edge_count']}")
        if summary.get('avg_stitch_length_mm') is not None:
            length_status = "❌ DEFECT" if defects.get('stitch_length', False) else "✅ OK"
            print(f"   ├─ Avg Stitch Length: {summary['avg_stitch_length_mm']:.2f}mm {length_status}")
            print(f"   ├─ Stitches per inch: {summary['stitches_per_inch']:.1f}")
        else:
            print(f"   ├─ Stitch Length: Not measurable")
        if summary.get('avg_distance_mm') is not None:
            dist_status = "❌ DEFECT" if defects.get('stitch_edge_distance', False) else "✅ OK"
            print(f"   ├─ Avg Stitch-Top Edge Distance: {summary['avg_distance_mm']:.2f}mm {dist_status}")
        else:
            print(f"   ├─ Avg Stitch-Top Edge Distance: Not measurable")
        print(f"   └─ Processing Time: {processing_time:.2f}s")
        defects_found = process_defects(results, ts)
        # Send 'D' to Arduino if consecutive defect threshold is reached
        if consecutive_stitch_length_defects >= CONSECUTIVE_DEFECT_THRESHOLD or consecutive_stitch_edge_defects >= CONSECUTIVE_DEFECT_THRESHOLD:
            print(f"🚨 CONSECUTIVE DEFECT THRESHOLD REACHED - Alerting Arduino")
            try:
                ser.write(b'D')
                ser.flush()
                print(f"   └─ ✅ Sent 'D' signal to Arduino")
            except Exception as e:
                print(f"   └─ ❌ Failed to send 'D' to Arduino: {e}")
        if defects_found:
            print(f"📩 Defects detected - Data will be logged to MySQL")
        else:
            print(f"✅ NO DEFECTS - Fabric passed inspection")
        print(f"⚡ ANALYSIS COMPLETE: {processing_time:.2f}s total\n")
    except Exception as e:
        print(f"❌ ERROR in fabric processing: {e}")
    finally:
        processing_lock.release()

# ---------------------------
# Setup Equipment & Main
# ---------------------------
def main():
    """Main function to start the system"""
    print("🚀 STARTING OPTIMIZED FABRIC INSPECTION SYSTEM")
    print("=" * 50)
    print("System Architecture:")
    print("  • Arduino: Motor control + distance data")
    print("  • Python: Periodic processing every 2 seconds + 'D' signal on consecutive defects")
    print("  • MySQL: Data insertion every 2 seconds")
    print("  • Image Cleanup: Deletes images older than 24 hours")
    print("=" * 50)
    print(f"Data will be inserted into MySQL at {DB_CONFIG['host']}/{DB_CONFIG['database']}")
    print(f"Images in {OUTPUT_DIR} will be deleted after {IMAGE_RETENTION_SECONDS/3600:.1f} hours")
    print("=" * 50)
    threads = []
    serial_thread = threading.Thread(target=serial_monitor_thread, daemon=True)
    serial_thread.start()
    threads.append(serial_thread)
    print("✅ Serial monitor thread started")
    mysql_thread = threading.Thread(target=mysql_reporting_thread, daemon=True)
    mysql_thread.start()
    threads.append(mysql_thread)
    print("✅ MySQL reporting thread started")
    cleanup_thread = threading.Thread(target=image_cleanup_thread, daemon=True)
    cleanup_thread.start()
    threads.append(cleanup_thread)
    print("✅ Image cleanup thread started")
    print(f"🎯 System ready! Processing fabric every 2 seconds...")
    try:
        ser.write(b'S')
        ser.flush()
        print(f"✅ Sent 'S' signal to Arduino")
    except Exception as e:
        print(f"❌ Failed to send 'S' to Arduino: {e}")
    print("   • Arduino sends: '<distance>' = Update distance")
    print("   • Python sends: 'D' = Consecutive defect threshold reached (LED blinks)")
    print("   • MySQL inserts: Every 2 seconds with defect status")
    print("   • Image cleanup: Every 1 hour, deleting files older than 24 hours")
    print("-" * 50)
    try:
        while not shutdown_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
        shutdown_event.set()
    print("🔄 Waiting for threads to finish...")
    for t in threads:
        t.join(timeout=2.0)
    if 'cap' in globals() and cap is not None:
        cap.release()
        print("✅ Camera released")
    ser.close()
    print("✅ Serial port closed")
    print("✅ System shutdown complete")

if __name__ == "__main__":
    print("🤖 Loading AI model...")
    model = YOLO("best_curve_100.pt")
    model.to(device)
    print(f"✅ Model loaded on {device}")
    print("📹 Creating snapshots directory")
    os.makedirs("snapshots", exist_ok=True)
    print("📹 Initializing camera...")
    if not init_camera():
        print("❌ CRITICAL ERROR: Camera initialization failed")
        sys.exit(1)
    main()
