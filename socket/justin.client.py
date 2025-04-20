import socket
import cv2
import pickle
import struct
import threading
import queue
import time
import json
import sys
import os
import numpy as np

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO  # Uncomment when model is available
from object_detection.yolo import get_bounding_boxes, getColours

class VideoStreamClient:
    def __init__(self, host=None, port=9999, buffer_size=10):
        self.host = host
        self.port = port
        self.control_host = host  # Same as video stream by default
        self.control_port = 9090  # Default control port
        self.client_socket = None
        self.control_socket = None
        self.running = False
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.current_frame = None
        self.current_sensor_data = {}
        self.detection_model = None
        self.detection_in_progress = False
        self.current_processed_package = None
        self.processed_frames_count = 0
        self.processing_start_time = time.time()
        self.processing_fps = 0
        self.frame_buffer_with_ids = {}  # Store frames with unique IDs
        self.processed_results = {}  # Store processed results by frame ID
        self.next_frame_id = 0
        
        # Box tracking and stabilization
        self.detection_history = {}  # Track objects by ID
        self.history_length = 3      # Number of frames to keep in history
        self.min_detection_frames = 2  # Minimum frames an object must be detected to display
        self.process_every_n_frames = 1  # Only process every N frames
        self.frame_counter = 0
        
        # Performance settings
        self.resize_for_detection = True  # Resize frames before detection for better performance
        self.detection_width = 320  # Smaller width for faster detection
        self.original_size = None   # Store original frame size
        
        # Network optimization settings
        self.recv_buffer_size = 131072  # 128KB buffer for socket operations (increased from 64KB)
        self.reception_fps = 0         # Track reception frame rate
        self.reception_frame_count = 0
        self.reception_start_time = time.time()
        self.current_jpeg_quality = 0  # Track quality reported by server
        self.last_packet_size = 0      # Track last packet size
        
        # Thread synchronization
        self.frame_ready_event = threading.Event()
        
    def connect_to_server(self):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if self.host is None:
                self.host = input("Enter server IP address: ")
            print(f"Connecting to server at {self.host}:{self.port}")
            self.client_socket.connect((self.host, self.port))
            print("Connected to server successfully")
            self.running = True
            self.reception_start_time = time.time()
            return True
        except Exception as e:
            print(f"Error connecting to server: {e}")
            self.cleanup()
            return False

    def listen_for_controls(self):
        try:
            self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.control_socket.connect((self.control_host, self.control_port))
            print(f"Connected to control server at {self.control_host}:{self.control_port}")

            buffer = b""
            while self.running:
                data = self.control_socket.recv(1024)
                if not data:
                    break
                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    try:
                        msg = json.loads(line.decode('utf-8'))
                        self.current_sensor_data.update({
                            "steering": msg.get("steering"),
                            "speed": msg.get("speed")
                        })
                        print(f"Received control: {self.current_sensor_data}")
                    except json.JSONDecodeError:
                        print("Invalid control message received.")
        except Exception as e:
            print(f"Control connection error: {e}")
        finally:
            if self.control_socket:
                self.control_socket.close()
                print("Control socket closed")

    def cleanup(self):
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        if self.control_socket:
            try:
                self.control_socket.close()
            except:
                pass
            self.control_socket = None
        cv2.destroyAllWindows()
        print("Client resources cleaned up")

    def start(self):
        if not self.connect_to_server():
            return

        try:
            print("Loading YOLO model...")
            model_path = 'object_detection/yoloe-11s-seg-pf.pt'
            self.detection_model = YOLO(model_path)
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Object detection will be disabled")

        reception_thread = threading.Thread(target=self.receive_frames, daemon=True)
        reception_thread.start()

        processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        processing_thread.start()

        self.control_thread = threading.Thread(target=self.listen_for_controls, daemon=True)
        self.control_thread.start()

        self.display_frames()

    
    def decompress_frame(self, compressed_frame):
        """Decompress JPEG frame data back to numpy array"""
        return cv2.imdecode(np.frombuffer(compressed_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    def decompress_frame_with_roi(self, compressed_data):
        """Decompress frame with ROI overlay for higher quality center region"""
        # Extract components
        full_frame_data = compressed_data['full_frame']
        roi_data = compressed_data['roi']
        roi_x, roi_y, roi_w, roi_h = compressed_data['roi_pos']
        
        # Decompress full frame (lower quality)
        full_frame = self.decompress_frame(full_frame_data)
        
        # Decompress ROI (higher quality)
        roi = self.decompress_frame(roi_data)
        
        # Overlay ROI on full frame
        if full_frame is not None and roi is not None:
            full_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = roi
            
        return full_frame
    
    def receive_frames(self):
        """Receive and buffer video frames from the server (producer)"""
        try:
            data = b""
            payload_size = struct.calcsize("L")
            
            while self.running:
                while len(data) < payload_size:
                    packet = self.client_socket.recv(self.recv_buffer_size)
                    if not packet:
                        self.running = False
                        break
                    data += packet
                
                if not self.running:
                    break
                
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]
                
                while len(data) < msg_size:
                    data += self.client_socket.recv(self.recv_buffer_size)
                
                # Extract the frame
                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                # Deserialize the frame
                frame = pickle.loads(frame_data)
                
                # Update current frame (latest frame always available)
                self.current_frame = frame.copy()
                
                try:
                    if not self.frame_buffer.full():
                        # Store the frame with its sensor data
                        frame_id = self.next_frame_id
                        self.next_frame_id += 1
                        self.frame_buffer_with_ids[frame_id] = {'frame': frame, 'sensor_data': self.current_sensor_data}
                        self.frame_buffer.put_nowait(self.frame_buffer_with_ids[frame_id])
                except:
                    pass
                
                # Update reception FPS metrics
                self.reception_frame_count += 1
                if self.reception_frame_count % 30 == 0:
                    elapsed = time.time() - self.reception_start_time
                    if elapsed > 0:
                        self.reception_fps = self.reception_frame_count / elapsed
                        # Reset counters every 300 frames for a more recent average
                        if self.reception_frame_count >= 300:
                            self.reception_frame_count = 0
                            self.reception_start_time = time.time()
                
        except ConnectionResetError:
            print("Server disconnected")
        except Exception as e:
            print(f"Error receiving frames: {e}")
        finally:
            self.running = False
    
    def process_frames(self):
        """Process frames with object detection (consumer)"""
        while self.running:
            self.frame_counter += 1
            
            # Only process every Nth frame to reduce CPU load and add stability
            if self.detection_model and not self.detection_in_progress and self.frame_counter % self.process_every_n_frames == 0:
                try:
                    # Get a frame for processing
                    frame = None
                    
                    # Prefer current_frame over buffered frames for most up-to-date processing
                    if self.current_frame is not None:
                        frame = self.current_frame.copy()
                    elif not self.frame_buffer.empty():
                        # If no new frame but buffer has frames, use from buffer
                        frame_data = self.frame_buffer.get()
                    
                    if frame_data is not None:
                        self.detection_in_progress = True
                        self.process_frame_with_detection(frame_data)
                        self.detection_in_progress = False
                        
                    # Update processing FPS counter
                    self.processed_frames_count += 1
                    elapsed = time.time() - self.processing_start_time
                    if elapsed >= 1.0:
                        self.processing_fps = self.processed_frames_count / elapsed
                        print(f"Processing rate: {self.processing_fps:.2f} FPS")
                        self.processed_frames_count = 0
                        self.processing_start_time = time.time()
                except Exception as e:
                    print(f"Error in frame processing: {e}")
                    self.detection_in_progress = False
            time.sleep(0.01)
    
    def process_frame_with_detection(self, frame_data):
        """Apply object detection to a frame"""
        
        print('Processing frame with detection')
        return
        try:
            # Perform object detection
            results = self.detection_model.predict(frame, conf=0.25)
            
            # Process the results and draw bounding boxes
            processed_frame = frame.copy()
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Get class name
                    class_name = result.names[cls]
                    
                    # Draw on the processed frame
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(processed_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Update the current frame with the processed version
            self.current_frame = processed_frame
            
            # Add your custom processing logic here
            # For example, logging detections, sending commands based on detections, etc.
            
        except Exception as e:
            print(f"Error in object detection: {e}")
    
    def display_frames(self):
        """Display received and processed frames"""
        try:
            cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
            last_displayed_timestamp = 0
            display_fps = 0
            fps_start_time = time.time()
            frame_count = 0
            last_smoothed_boxes = []  # Cache last valid set of boxes
            last_boxes_time = 0
            
            while self.running:
                if self.current_frame is not None:
                    # Display the current frame (original or processed)
                    cv2.imshow('Stream', self.current_frame)
                    
                    # Check for 'q' key press to quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.running = False
                        break
                    
                    # Adaptive sleep to maintain target frame rate
                    frame_time = time.time() - frame_start
                    target_frame_time = 1.0 / 30.0  # target 30fps
                    if frame_time < target_frame_time:
                        time.sleep(target_frame_time - frame_time)
                else:
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("Display stopped by user")
        except Exception as e:
            print(f"Error displaying frames: {e}")
        finally:
            self.cleanup()
    
    def draw_smoothed_boxes(self, frame, smoothed_boxes):
        """Draw smoothed bounding boxes on the frame"""
        for box in smoothed_boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box['xyxy']]
            cls = box['cls']
            conf = box['conf']
            obj_id = box['id']
            
            # Get class name and color
            try:
                # Get class name from the YOLO model's names dictionary
                class_name = self.detection_model.names[cls]
            except:
                class_name = f"Class {cls}"
                
            color = getColours(cls)
            
            # Draw rectangle with slightly increased thickness for stability
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Reduced thickness for performance
            
            # Simplified label to reduce text rendering overhead
            label = f'{class_name[:3]}{conf:.1f}'  # Shorter label
            cv2.putText(frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # Smaller, thinner text
        
        return frame

    def draw_detection_results(self, frame, results):
        """Draw detection boxes directly on a frame (used with raw results)"""
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                try:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    class_name = result.names[cls]
                    color = getColours(cls)
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except:
                    continue
                
        return frame

    def add_sensor_data_to_frame(self, frame, sensor_data):
        """Add sensor data overlay to a frame"""
        ultrasonic = sensor_data.get('ultrasonic')
        
        # Only display ultrasonic data to reduce rendering overhead
        if ultrasonic is not None:
            cv2.putText(frame, f"Dist: {ultrasonic:.1f}cm", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Simpler, faster text
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        cv2.destroyAllWindows()
        print("Client resources cleaned up")

def main():
    client = VideoStreamClient()
    client.start()

if __name__ == "__main__":
    main()
