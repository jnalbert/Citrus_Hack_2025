import socket
import cv2
import pickle
import struct
import threading
import queue
import time
import json

# from ultralytics import YOLO  # Uncomment when model is available

class VideoStreamClient:
    def __init__(self, host=None, port=8080, control_port=9090, buffer_size=10):
        """
        Initialize the video streaming client
        
        Args:
            host (str): Host IP to connect to. If None, will prompt user
            port (int): Port to connect to for video stream
            control_port (int): Port to listen for control commands
            buffer_size (int): Max number of frames to buffer
        """
        self.host = host
        self.port = port
        self.control_port = control_port
        self.client_socket = None
        self.running = False
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.current_frame = None
        self.detection_model = None
        self.detection_in_progress = False
        self.control_thread = None
        
    def connect_to_server(self):
        """Connect to the video streaming server"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if self.host is None:
                self.host = input("Enter server IP address: ")
            print(f"Connecting to server at {self.host}:{self.port}")
            self.client_socket.connect((self.host, self.port))
            print("Connected to server successfully")
            self.running = True
            return True
        except Exception as e:
            print(f"Error connecting to server: {e}")
            self.cleanup()
            return False
    
    def start(self):
        """Start the client with frame reception and processing threads"""
        if not self.connect_to_server():
            return
        
        # Initialize YOLO model (optional)
        try:
            print("Loading YOLO model...")
            # self.detection_model = YOLO('object_detection/yolov8n.pt')
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Object detection will be disabled")
        
        # Start frame reception thread
        reception_thread = threading.Thread(target=self.receive_frames, daemon=True)
        reception_thread.start()
        
        # Start frame processing thread
        processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        processing_thread.start()

        # Start control command listener
        self.control_thread = threading.Thread(target=self.listen_for_controls, daemon=True)
        self.control_thread.start()
        
        # Display frames in main thread
        self.display_frames()
    
    def receive_frames(self):
        """Receive and buffer video frames from the server (producer)"""
        try:
            data = b""
            payload_size = struct.calcsize("L")
            
            while self.running:
                while len(data) < payload_size:
                    packet = self.client_socket.recv(4096)
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
                    data += self.client_socket.recv(4096)
                
                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                frame = pickle.loads(frame_data)
                self.current_frame = frame.copy()
                
                try:
                    if not self.frame_buffer.full():
                        self.frame_buffer.put_nowait(frame)
                except:
                    pass
                
        except ConnectionResetError:
            print("Server disconnected")
        except Exception as e:
            print(f"Error receiving frames: {e}")
        finally:
            self.running = False
    
    def process_frames(self):
        """Process frames with object detection (consumer)"""
        while self.running:
            if self.detection_model and not self.detection_in_progress:
                try:
                    frame = None
                    if self.current_frame is not None:
                        frame = self.current_frame.copy()
                    elif not self.frame_buffer.empty():
                        frame = self.frame_buffer.get()
                    
                    if frame is not None:
                        self.detection_in_progress = True
                        self.process_frame_with_detection(frame)
                        self.detection_in_progress = False
                except Exception as e:
                    print(f"Error in frame processing: {e}")
                    self.detection_in_progress = False
            time.sleep(0.01)
    
    def process_frame_with_detection(self, frame):
        """Apply object detection to a frame"""
        print('Processing frame with detection')
        return
        try:
            results = self.detection_model.predict(frame, conf=0.25)
            processed_frame = frame.copy()
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = result.names[cls]
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(processed_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.current_frame = processed_frame
        except Exception as e:
            print(f"Error in object detection: {e}")
    
    def listen_for_controls(self):
        """Listen for control commands like steering and speed"""
        control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        control_socket.bind(('', self.control_port))
        control_socket.listen(1)
        print(f"[CONTROL] Listening for control commands on port {self.control_port}")

        conn, addr = control_socket.accept()
        print(f"[CONTROL] Connection from {addr}")

        try:
            while self.running:
                data = conn.recv(1024)
                if not data:
                    break
                try:
                    action = json.loads(data.decode('utf-8'))
                    print(f"[CONTROL] Received action: {action}")
                    # TODO: Control robot motors using action['steering'] and action['speed']
                except json.JSONDecodeError as e:
                    print(f"[CONTROL] JSON decode error: {e}")
        except Exception as e:
            print(f"[CONTROL] Error: {e}")
        finally:
            conn.close()
            control_socket.close()
    
    def display_frames(self):
        """Display received and processed frames"""
        try:
            while self.running:
                if self.current_frame is not None:
                    cv2.imshow('Stream', self.current_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.running = False
                        break
                else:
                    time.sleep(0.01)
        except KeyboardInterrupt:
            print("Display stopped by user")
        except Exception as e:
            print(f"Error displaying frames: {e}")
        finally:
            self.cleanup()
    
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
