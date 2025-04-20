import socket
import cv2
import pickle
import struct
import sys
import time
import numpy as np
from picamera2 import Picamera2
from picarx import Picarx

class VideoStreamServer:
    def __init__(self, host='0.0.0.0', port=8080):
        """
        Initialize the video streaming server
        
        Args:
            host (str): Host IP to bind to. Default '0.0.0.0' (all interfaces)
            port (int): Port to bind to. Default 8080
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.video_capture = None
        self.running = False
        self.px = None
        
        # Performance optimization settings
        self.frame_count = 0
        self.start_time = 0
        self.frame_interval = 0.04  # Target 25 FPS (40ms between frames)
        self.jpeg_quality = 80  # Compression quality (0-100)
        self.resolution = (320, 240)  # Lower resolution for faster transmission
        self.send_buffer_size = 1000000  # 1MB buffer for socket operations
        self.monitor_interval = 30  # Print stats every 30 frames
    
    def get_local_ip(self):
        """Get the local IP address that others can connect to."""
        try:
            # Create a socket that connects to an external server
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Doesn't actually connect, just sets up the socket
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            # Fallback method
            return socket.gethostbyname(socket.gethostname())
    
    def setup_server(self):
        """Initialize and bind the server socket"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Allow reuse of the address
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Set socket buffer size to optimize for streaming
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.send_buffer_size)
            
            # Bind the socket to the address
            self.server_socket.bind((self.host, self.port))
            
            # Start listening (queue up to 5 connection requests)
            self.server_socket.listen(5)
            
            # Display connection information
            local_ip = self.get_local_ip()
            print(f"Server started on {local_ip}:{self.port}")
            print(f"Clients should connect to this address")
            
            self.running = True
            self.px = Picarx()
            return True
        
        except Exception as e:
            print(f"Error setting up server: {e}")
            self.cleanup()
            return False
    
    def start_streaming(self):
        """Accept connections and stream video to clients"""
        if not self.setup_server():
            return
        
        try:
            print("Waiting for client connection...")
            
            while self.running:
                # Accept client connection
                self.client_socket, client_address = self.server_socket.accept()
                print(f"Connected to client at {client_address}")
                
                # Initialize Picamera2 with optimized settings
                self.video_capture = Picamera2()
                config = self.video_capture.create_preview_configuration(
                    main={"size": self.resolution, "format": "RGB888"}
                )
                self.video_capture.configure(config)
                self.video_capture.start()
                
                # Wait a moment for camera to initialize
                time.sleep(0.5)
                
                # Stream video to the client
                self.stream_to_client()
                
                # Clean up client resources
                if self.client_socket:
                    self.client_socket.close()
                    self.client_socket = None
                
                if self.video_capture:
                    self.video_capture.stop()
                    self.video_capture = None
        
        except KeyboardInterrupt:
            print("\nServer stopped by user")
        except Exception as e:
            print(f"Error in streaming: {e}")
        finally:
            self.cleanup()
    
    def compress_frame(self, frame):
        """Compress frame to JPEG format for efficient transmission"""
        _, compressed = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return compressed
    
    def stream_to_client(self):
        """Stream video frames to the connected client"""
        try:
            self.start_time = time.time()
            self.frame_count = 0
            last_frame_time = 0
            last_sensor_read_time = 0
            sensor_read_interval = 0.1  # Read sensors every 100ms
            ultrasonic_data = 0  # Initial value
            
            while self.running and self.client_socket:
                current_time = time.time()
                
                # Control frame rate by limiting how often we send frames
                if current_time - last_frame_time < self.frame_interval:
                    # Skip this iteration to maintain target frame rate
                    time.sleep(0.001)  # Short sleep to prevent CPU hogging
                    continue
                
                # Read a frame from the Picamera2
                frame = self.video_capture.capture_array()
                
                # Get sensor data (less frequently than frames)
                if current_time - last_sensor_read_time > sensor_read_interval:
                    ultrasonic_data = self.px.ultrasonic.read()
                    last_sensor_read_time = current_time
                
                # Convert from BGR to RGB if needed (depends on camera output)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Compress the frame for faster transmission
                compressed_frame = self.compress_frame(frame)
                
                # Create a data package containing both compressed frame and sensor data
                data_package = {
                    'frame_compressed': compressed_frame,
                    'ultrasonic': ultrasonic_data,
                    'timestamp': current_time,
                    'frame_count': self.frame_count
                }
                
                # Serialize the data package using pickle with the highest protocol for efficiency
                serialized_data = pickle.dumps(data_package, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Create message with package size and package data
                message = struct.pack("L", len(serialized_data)) + serialized_data
                
                # Send the data package to the client
                self.client_socket.sendall(message)
                
                # Update timing information
                last_frame_time = current_time
                self.frame_count += 1
                
                # Print FPS every few frames
                if self.frame_count % self.monitor_interval == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed
                    print(f"Streaming at {fps:.2f} FPS, frame size: {len(serialized_data)/1024:.1f} KB")
                
        except ConnectionResetError:
            print("Client disconnected")
        except Exception as e:
            print(f"Error streaming to client: {e}")
    
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
        
        # Close the server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None
        
        # Release the video capture
        if self.video_capture:
            if hasattr(self.video_capture, 'stop'):
                self.video_capture.stop()
            self.video_capture = None
        
        print("Server resources cleaned up")

def main():
    # Create server with default host and port
    server = VideoStreamServer()
    
    # Start streaming
    server.start_streaming()

if __name__ == "__main__":
    main()

