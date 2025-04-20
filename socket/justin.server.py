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
        self.frame_interval = 0.03  # Target 30+ FPS (33ms between frames)
        self.jpeg_quality = 70  # Slightly lower quality for faster transmission (was 80)
        self.resolution = (320, 240)  # Keep resolution the same
        self.send_buffer_size = 1000000  # 1MB buffer for socket operations
        self.monitor_interval = 30  # Print stats every 30 frames
        
        # Adaptive parameters
        self.min_frame_interval = 0.01  # Maximum 100 FPS
        self.max_frame_interval = 0.1   # Minimum 10 FPS
        self.target_packet_size = 20    # Target 20KB packet size
        self.adaptive_quality = True    # Enable adaptive quality
        self.min_jpeg_quality = 40      # Minimum quality
        self.max_jpeg_quality = 85      # Maximum quality
        
        # ROI-based compression (higher quality in center, lower on edges)
        self.use_roi_compression = False  # Experimental feature - enable if needed
    
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
            
            # Set TCP_NODELAY to reduce latency (disable Nagle's algorithm)
            self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
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
                
                # Configure client socket for performance
                self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
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
    
    def compress_frame(self, frame, quality=None):
        """Compress frame to JPEG format for efficient transmission"""
        # Use provided quality or instance default
        jpeg_quality = quality if quality is not None else self.jpeg_quality
        _, compressed = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        return compressed
    
    def compress_frame_with_roi(self, frame):
        """Compress frame with region of interest - higher quality in center"""
        h, w = frame.shape[:2]
        
        # Define center region (50% of frame)
        center_w, center_h = w // 2, h // 2
        roi_w, roi_h = w // 2, h // 2
        
        # Extract center ROI
        roi = frame[center_h - roi_h//2:center_h + roi_h//2, 
                   center_w - roi_w//2:center_w + roi_w//2]
        
        # Compress ROI with higher quality
        _, roi_compressed = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, self.max_jpeg_quality])
        
        # Compress full frame with lower quality
        _, frame_compressed = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.min_jpeg_quality])
        
        # Return both for client to reconstruct
        return {
            'full_frame': frame_compressed,
            'roi': roi_compressed,
            'roi_pos': (center_w - roi_w//2, center_h - roi_h//2, roi_w, roi_h)
        }
    
    def adjust_parameters(self, packet_size_kb):
        """Dynamically adjust compression and timing parameters based on network conditions"""
        # Adjust frame interval (FPS) based on packet size
        if packet_size_kb > self.target_packet_size * 1.5:
            # Network is congested, slow down
            self.frame_interval = min(self.frame_interval * 1.05, self.max_frame_interval)
        elif packet_size_kb < self.target_packet_size * 0.8:
            # Network has capacity, speed up
            self.frame_interval = max(self.frame_interval * 0.95, self.min_frame_interval)
        
        # Adjust JPEG quality based on packet size if adaptive quality is enabled
        if self.adaptive_quality:
            if packet_size_kb > self.target_packet_size * 1.2:
                # Reduce quality to decrease packet size
                self.jpeg_quality = max(self.jpeg_quality - 2, self.min_jpeg_quality)
            elif packet_size_kb < self.target_packet_size * 0.8:
                # Increase quality since network has capacity
                self.jpeg_quality = min(self.jpeg_quality + 1, self.max_jpeg_quality)
    
    def stream_to_client(self):
        """Stream video frames to the connected client"""
        try:
            self.start_time = time.time()
            self.frame_count = 0
            last_frame_time = 0
            last_sensor_read_time = 0
            sensor_read_interval = 0.2  # Read sensors less frequently (200ms)
            ultrasonic_data = 0  # Initial value
            last_packet_size = 0
            
            # Keep track of recent packet sizes for adaptive adjustments
            recent_packet_sizes = []
            
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
                    try:
                        ultrasonic_data = self.px.ultrasonic.read()
                        last_sensor_read_time = current_time
                    except:
                        # If sensor read fails, just continue with last value
                        pass
                
                # Convert from BGR to RGB (if needed)
                if frame.shape[2] == 3:  # Only if it's a color image
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Choose compression method
                if self.use_roi_compression:
                    compressed_data = self.compress_frame_with_roi(frame)
                    data_package = {
                        'frame_compressed_roi': compressed_data,
                        'ultrasonic': ultrasonic_data,
                        'timestamp': current_time,
                        'frame_count': self.frame_count,
                        'jpeg_quality': self.jpeg_quality
                    }
                else:
                    # Regular compression
                    compressed_frame = self.compress_frame(frame)
                    data_package = {
                        'frame_compressed': compressed_frame,
                        'ultrasonic': ultrasonic_data,
                        'timestamp': current_time,
                        'frame_count': self.frame_count,
                        'jpeg_quality': self.jpeg_quality
                    }
                
                # Serialize the data package using pickle with the highest protocol
                serialized_data = pickle.dumps(data_package, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Create message with package size and package data
                message = struct.pack("L", len(serialized_data)) + serialized_data
                
                # Calculate packet size in KB
                packet_size_kb = len(serialized_data) / 1024
                
                # Keep track of recent packet sizes (up to 10)
                recent_packet_sizes.append(packet_size_kb)
                if len(recent_packet_sizes) > 10:
                    recent_packet_sizes.pop(0)
                
                # Adjust parameters every 5 frames based on average recent packet size
                if self.frame_count % 5 == 0 and recent_packet_sizes:
                    avg_packet_size = sum(recent_packet_sizes) / len(recent_packet_sizes)
                    self.adjust_parameters(avg_packet_size)
                
                # Send the data package to the client
                try:
                    self.client_socket.sendall(message)
                    last_packet_size = packet_size_kb
                except:
                    print("Error sending frame, client may have disconnected")
                    break
                
                # Update timing information
                last_frame_time = current_time
                self.frame_count += 1
                
                # Print FPS and stats every few frames
                if self.frame_count % self.monitor_interval == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed
                    print(f"Streaming at {fps:.2f} FPS, frame size: {packet_size_kb:.1f} KB, quality: {self.jpeg_quality}, interval: {self.frame_interval*1000:.1f}ms")
                
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

