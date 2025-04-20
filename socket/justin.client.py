import socket
import cv2
import pickle
import struct
import threading
import queue
import time
from ultralytics import YOLO  # Assuming using YOLO for object detection

class VideoStreamClient:
    def __init__(self, host=None, port=9999, buffer_size=10):
        """
        Initialize the video streaming client
        
        Args:
            host (str): Host IP to connect to. If None, will prompt user
            port (int): Port to connect to. Default 9999
            buffer_size (int): Max number of frames to buffer
        """
        self.host = host
        self.port = port
        self.client_socket = None
        self.running = False
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.current_frame = None
        self.detection_model = None
        self.detection_in_progress = False
        
    def connect_to_server(self):
        """Connect to the video streaming server"""
        try:
            # Create socket
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Get server address if not provided
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
        
        # Initialize YOLO model
        try:
            print("Loading YOLO model...")
            # self.detection_model = YOLO('object_detection/yolov8n.pt')
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
    
    def receive_frames(self):
        """Receive and buffer video frames from the server (producer)"""
        try:
            data = b""
            payload_size = struct.calcsize("L")
            
            while self.running:
                # Receive data until we have the payload size
                while len(data) < payload_size:
                    packet = self.client_socket.recv(4096)
                    if not packet:
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
                    data += self.client_socket.recv(4096)
                
                # Extract the frame
                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                # Deserialize the frame
                frame = pickle.loads(frame_data)
                
                # Update current frame (latest frame always available)
                self.current_frame = frame.copy()
                
                # Add to buffer if space available (non-blocking)
                try:
                    if not self.frame_buffer.full():
                        self.frame_buffer.put_nowait(frame)
                except:
                    pass  # Skip frame if buffer is full
                
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
                    # Get a frame for processing
                    frame = None
                    
                    # Prefer current_frame over buffered frames for most up-to-date processing
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
            
            # Sleep a bit to prevent CPU overuse
            time.sleep(0.01)
    
    def process_frame_with_detection(self, frame):
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
            while self.running:
                if self.current_frame is not None:
                    # Display the current frame (original or processed)
                    cv2.imshow('Stream', self.current_frame)
                    
                    # Check for 'q' key press to quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.running = False
                        break
                else:
                    # Sleep briefly if no frame is available
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

def main():
    # Create client
    client = VideoStreamClient()
    
    # Start receiving and processing
    client.start()

if __name__ == "__main__":
    main()