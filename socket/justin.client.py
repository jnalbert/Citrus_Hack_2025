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
from ultralytics import YOLO  # Assuming using YOLO for object detection
from object_detection.yolo import get_bounding_boxes, getColours
from navigation.navigation import generate_action_from_bounding_boxes

class VideoStreamClient:
    def __init__(self, host=None, port=8080, buffer_size=10):
        """
        Initialize the video streaming client
        
        Args:
            host (str): Host IP to connect to. If None, will prompt user
            port (int): Port to connect to. Default 8080
            buffer_size (int): Max number of frames to buffer
        """
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
        self.history_length = 2      # Number of frames to keep in history
        self.min_detection_frames = 1  # Minimum frames an object must be detected to display
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
        
    def connect_to_server(self):
        """Connect to the video streaming server"""
        try:
            # Create socket
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Set socket options for better streaming performance
            self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.recv_buffer_size)
            
            # Disable Nagle's algorithm to reduce latency
            self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # Get server address if not provided
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
    
    def start(self):
        """Start the client with frame reception and processing threads"""
        if not self.connect_to_server():
            return
        
        # Connect to control socket
        try:
            self.control_host = self.host  # Use same host as video stream
            print(f"Connecting to control server at {self.control_host}:{self.control_port}")
            self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.control_socket.connect((self.control_host, self.control_port))
            
            # Set TCP_NODELAY to reduce latency
            self.control_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # Start thread to listen for control messages
            control_thread = threading.Thread(target=self.listen_for_controls)
            control_thread.daemon = True
            control_thread.start()
            
            print("Connected to control server successfully")
        except Exception as e:
            print(f"Warning: Could not connect to control server: {e}")
            print("Object detection results will not be sent back to server")
        
        # Initialize YOLO model - use smaller model for performance
        try:
            print("Loading YOLO model...")
            # Use a smaller and faster model for better performance
            model_path = 'object_detection/yoloe-11s-seg-pf.pt'
            
            # Try CPU if MPS is slow, or try CUDA if available
            self.detection_model = YOLO(model_path)
            
            # Optional: Use export model for even faster inference
            # self.detection_model.export(format='onnx')  # Export once
            # self.detection_model = YOLO('object_detection/yoloe-11s-seg-pf.onnx')  # Use exported model
            
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Object detection will be disabled")
        
        # Start frame reception thread
        reception_thread = threading.Thread(target=self.receive_frames)
        reception_thread.daemon = True
        reception_thread.start()
        
        # Start frame processing thread
        processing_thread = threading.Thread(target=self.process_frames)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Main thread displays frames
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
                # Receive data until we have the payload size
                while len(data) < payload_size:
                    packet = self.client_socket.recv(self.recv_buffer_size)
                    if not packet:
                        print("Server closed connection")
                        self.running = False
                        break
                    data += packet
                
                if not self.running:
                    break
                
                # Extract the frame size
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]
                
                # Receive the frame data
                while len(data) < msg_size:
                    data += self.client_socket.recv(self.recv_buffer_size)
                
                # Extract the data package
                package_data = data[:msg_size]
                data = data[msg_size:]
                
                # Track packet size
                self.last_packet_size = len(package_data) / 1024  # KB
                
                # Measure time for decompression
                decomp_start = time.time()
                
                # Deserialize the data package
                data_package = pickle.loads(package_data)
                
                # Store JPEG quality reported by server
                self.current_jpeg_quality = data_package.get('jpeg_quality', 0)
                
                # Extract and decompress frame using appropriate method
                frame = None
                
                # Handle ROI-based compression if available
                if 'frame_compressed_roi' in data_package:
                    compressed_data = data_package['frame_compressed_roi']
                    frame = self.decompress_frame_with_roi(compressed_data)
                # Handle standard compression
                elif 'frame_compressed' in data_package:
                    compressed_frame = data_package['frame_compressed']
                    frame = self.decompress_frame(compressed_frame)
                # Fallback for backward compatibility
                else:
                    frame = data_package.get('frame')
                
                decomp_time = (time.time() - decomp_start) * 1000  # ms
                
                if frame is None:
                    continue
                
                # Store original size if not already stored
                if self.original_size is None:
                    self.original_size = frame.shape[:2]  # Height, width
                
                # Store the sensor data
                self.current_sensor_data = {
                    'ultrasonic': data_package.get('ultrasonic', 0),
                    'timestamp': data_package.get('timestamp', time.time()),
                    'frame_count': data_package.get('frame_count', 0),
                    'jpeg_quality': self.current_jpeg_quality,
                    'packet_size_kb': self.last_packet_size,
                    'decomp_time_ms': decomp_time
                }
                
                # Update current frame (latest frame always available)
                self.current_frame = frame
                
                # Signal that a new frame is ready
                self.frame_ready_event.set()
                
                # Add to buffer if space available (non-blocking)
                try:
                    if not self.frame_buffer.full():
                        # Store the frame with its sensor data
                        frame_id = self.next_frame_id
                        self.next_frame_id += 1
                        self.frame_buffer_with_ids[frame_id] = {'frame': frame, 'sensor_data': self.current_sensor_data}
                        self.frame_buffer.put_nowait(self.frame_buffer_with_ids[frame_id])
                except:
                    pass  # Skip frame if buffer is full
                
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
                
        except socket.error as e:
            print(f"Socket error: {e}")
            self.running = False
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
                    frame_data = None
                    
                    # Wait for a new frame with timeout
                    if self.frame_ready_event.wait(timeout=0.1):
                        self.frame_ready_event.clear()  # Reset event
                        
                        # Prefer current_frame over buffered frames
                        if self.current_frame is not None:
                            frame_data = {
                                'frame': self.current_frame.copy(),
                                'sensor_data': self.current_sensor_data
                            }
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
            
            # Sleep a bit to prevent CPU overuse
            time.sleep(0.01)
    
    def process_frame_with_detection(self, frame_data):
        """Apply object detection to a frame"""
        
        original_frame = frame_data['frame']
        sensor_data = frame_data['sensor_data']
        
        try:
            # Process a copy of the frame
            detection_frame = original_frame.copy()
            
            # Resize frame for faster detection if enabled
            if self.resize_for_detection and detection_frame is not None:
                h, w = detection_frame.shape[:2]
                # Only resize if the frame is larger than our target size
                if w > self.detection_width:
                    # Calculate new height to maintain aspect ratio
                    new_h = int(h * (self.detection_width / w))
                    # Resize the frame for detection (smaller = faster)
                    detection_frame = cv2.resize(detection_frame, (self.detection_width, new_h))
            
            # Get detection results (faster on smaller frame)
            boxed_frame, results = get_bounding_boxes(detection_frame, self.detection_model, conf=0.6)
            
            # Update detection history and smooth bounding boxes
            self.update_detection_history(results, 
                                         original_size=original_frame.shape[:2] if self.resize_for_detection else None,
                                         detection_size=detection_frame.shape[:2] if self.resize_for_detection else None)
            
            # Create a package with the processed frame and results
            processed_package = {
                'original_frame': original_frame,
                'processed_frame': boxed_frame,
                'results': results,
                'sensor_data': sensor_data,
                'timestamp': time.time()
            }
            
            # Store this package (not just the frame)
            self.current_processed_package = processed_package
            
            # Store processed results for this frame ID
            frame_id = self.next_frame_id
            self.next_frame_id += 1
            self.frame_buffer_with_ids[frame_id] = {'frame': original_frame, 'sensor_data': sensor_data}
            self.processed_results[frame_id] = {'boxes': boxed_frame, 'results': results}
            
        except Exception as e:
            print(f"Error in object detection: {e}")
    
    def update_detection_history(self, results, original_size=None, detection_size=None):
        """Update detection history for tracking and smoothing"""
        # Extract tracked object IDs and bounding boxes
        detected_objects = {}
        
        # Calculate scale factors if resizing was used
        scale_x, scale_y = 1.0, 1.0
        if original_size is not None and detection_size is not None:
            scale_y = original_size[0] / detection_size[0]  # height ratio
            scale_x = original_size[1] / detection_size[1]  # width ratio
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                detected_objects["nothing"] = {
                            'xyxy': [0, 0, 0, 0],
                            'conf': 100,
                            'cls': 0,
                            'time': time.time()
                        }
                continue
            for box in boxes:
                try:
                    # Get box coordinates and tracking ID
                    xyxy = box.xyxy[0].tolist()
                    
                    # Scale coordinates back to original frame size if needed
                    if original_size is not None:
                        xyxy[0] *= scale_x  # x1
                        xyxy[2] *= scale_x  # x2
                        xyxy[1] *= scale_y  # y1
                        xyxy[3] *= scale_y  # y2
                    
                    # Use tracking ID if available, otherwise use a hash of the box location
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0])
                    else:
                        # Create a synthetic ID based on position and class
                        cls_id = int(box.cls[0])
                        center_x = int((xyxy[0] + xyxy[2]) / 2)
                        center_y = int((xyxy[1] + xyxy[3]) / 2)
                        # Use larger grid cells (100px) for more stable tracking
                        track_id = hash(f"{cls_id}_{center_x//100}_{center_y//100}") % 10000
                    
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Lower confidence threshold to maintain tracking through brief occlusions
                    if conf >= 0.3:
                        detected_objects[track_id] = {
                            'xyxy': xyxy,
                            'conf': conf,
                            'cls': cls,
                            'time': time.time()
                        }
                except:
                    continue
        
        # Update history with new detections
        current_time = time.time()
        for obj_id, obj_data in detected_objects.items():
            if obj_id not in self.detection_history:
                self.detection_history[obj_id] = []
            
            # Add new detection to history
            self.detection_history[obj_id].append(obj_data)
            
            # Keep only recent history
            if len(self.detection_history[obj_id]) > self.history_length:
                self.detection_history[obj_id].pop(0)
        
        # Remove old objects (not seen recently)
        ids_to_remove = []
        for obj_id, history in self.detection_history.items():
            if current_time - history[-1]['time'] > 1.0:  # 1 second timeout (faster cleanup)
                ids_to_remove.append(obj_id)
        
        for obj_id in ids_to_remove:
            del self.detection_history[obj_id]
    def get_smoothed_boxes(self):
        """Get temporally smoothed bounding boxes for display"""
        smoothed_boxes = []
        
        for obj_id, history in self.detection_history.items():
            # Only display objects with enough detection history
            if len(history) >= self.min_detection_frames:
                # Calculate weighted average of recent boxes
                box_sum = [0, 0, 0, 0]
                conf_sum = 0
                total_weight = 0
                
                # More recent detections get higher weight
                for i, detection in enumerate(history):
                    weight = (i + 1)  # Increasing weights for more recent detections
                    xyxy = detection['xyxy']
                    conf = detection['conf']
                    
                    for j in range(4):
                        box_sum[j] += xyxy[j] * weight
                    
                    conf_sum += conf * weight
                    total_weight += weight
                
                # Calculate weighted averages
                avg_box = [int(coord / total_weight) for coord in box_sum]
                avg_conf = conf_sum / total_weight
                
                # Use the most recent class and other data
                latest = history[-1]
                smoothed_boxes.append({
                    'id': obj_id,
                    'xyxy': avg_box,
                    'conf': avg_conf,
                    'cls': latest['cls']
                })
        
        return smoothed_boxes
    
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
            last_result_sent_time = 0  # Track when we last sent results
            
            while self.running:
                # Get the latest frame
                current_frame = self.current_frame
                
                if current_frame is not None:
                    frame_start = time.time()
                    
                    # Create a copy for display
                    display_frame = current_frame.copy()
                    
                    # Get smoothed detection boxes
                    smoothed_boxes = self.get_smoothed_boxes()
                    
                    # If no boxes found but we have recent boxes, use those
                    # This helps maintain display through momentary detection failures
                    if not smoothed_boxes and time.time() - last_boxes_time < 0.5:  # 500ms timeout
                        smoothed_boxes = last_smoothed_boxes
                    elif smoothed_boxes:
                        last_smoothed_boxes = smoothed_boxes
                        last_boxes_time = time.time()
                        
                        # Send detection results to server (limit frequency to avoid network congestion)
                        current_time = time.time()
                        if current_time - last_result_sent_time > 0.1:  # Send at most 10 times per second
                            # Calculate steering angle and speed based on detections
                            control_result = self.calculate_control_commands(smoothed_boxes)
                            
                            # Only send commands if we got a result (not None)
                            if control_result is not None:
                                steering_angle, speed = control_result
                                

                                self.send_control_commands(steering_angle, speed)
                    
                    # Draw boxes efficiently
                    self.draw_smoothed_boxes(display_frame, smoothed_boxes)
                    
                    # Add sensor data
                    if self.current_sensor_data:
                        self.add_sensor_data_to_frame(display_frame, self.current_sensor_data)
                    
                    # Calculate and display FPS (do this less frequently to save CPU)
                    frame_count += 1
                    elapsed = time.time() - fps_start_time
                    if elapsed >= 1.0:
                        display_fps = frame_count / elapsed
                        frame_count = 0
                        fps_start_time = time.time()
                    
                    # Display FPS and quality information in overlay
                    cv2.putText(
                        display_frame, 
                        f"Disp: {display_fps:.1f} FPS | Det: {self.processing_fps:.1f} FPS | Recv: {self.reception_fps:.1f} FPS | Q: {self.current_jpeg_quality}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                    )
                    
                    # Display network stats in second line
                    if 'packet_size_kb' in self.current_sensor_data and 'decomp_time_ms' in self.current_sensor_data:
                        kb = self.current_sensor_data['packet_size_kb']
                        ms = self.current_sensor_data['decomp_time_ms']
                        cv2.putText(
                            display_frame, 
                            f"Size: {kb:.1f} KB | Decomp: {ms:.1f} ms", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                        )
                    
                    # Display the frame - this is a potentially slow operation
                    cv2.imshow('Stream', display_frame)
                    
                    # Process UI events in batches for better performance
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
        
        # Close the client socket
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        print("Client resources cleaned up")

    def send_control_commands(self, steering_angle, speed):
        """
        Send control commands (steering angle and speed) back to the Raspberry Pi
        
        Args:
            steering_angle: The steering angle to send
            speed: The speed value to send
        """
        if not self.control_socket:
            print("No control socket connection available")
            return False
            
        try:
            # Create the control command message
            message = {
                "type": "control_commands",
                "timestamp": time.time(),
                "steering_angle": float(steering_angle),
                "speed": float(speed)
            }
            
            # Convert to JSON and send with newline terminator
            json_message = json.dumps(message) + "\n"
            self.control_socket.sendall(json_message.encode('utf-8'))
            return True
            
        except Exception as e:
            print(f"Error sending control commands: {e}")
            return False
    
    def calculate_control_commands(self, smoothed_boxes):
        """
        Calculate steering angle and speed based on detected objects
        
        Args:
            smoothed_boxes: List of detected objects
        
        Returns:
            Tuple of (steering_angle, speed) or None if not enough frames processed
        """
        bounding_boxes = []
        for box in smoothed_boxes:
            # Get class name from the detection model
            cls_id = box['cls']
            try:
                class_name = self.detection_model.names[cls_id]
            except:
                class_name = f"class_{cls_id}"  # Fallback if class name not found
                
            # Create the bounding box dictionary with proper label
            bounding_boxes.append({
                'label': class_name,
                'bbox': box['xyxy'],
                'confidence': box['conf']
            })
        
        # This will return None if not enough histograms have been collected
        action = generate_action_from_bounding_boxes(bounding_boxes)
        
        # Return None if the navigation system isn't ready yet
        if action is None:
          return None
        
        # Otherwise, return the computed steering and speed
        return action['steering'], action['speed']

def main():
    # Create client
    client = VideoStreamClient()
    
    # Start receiving and processing
    client.start()

if __name__ == "__main__":
    main()