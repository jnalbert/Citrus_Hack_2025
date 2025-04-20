import socket
import threading
import cv2
import pickle
import struct
import sys
import time
import json
import numpy as np

from movement.movement import MovementController

# Determine if running on Raspberry Pi or regular computer
IS_RASPBERRY_PI = True  # Default assumption
try:
    # Try to import Raspberry Pi specific libraries
    from picamera2 import Picamera2
    from picarx import Picarx
except ImportError:
    IS_RASPBERRY_PI = False
    print("Raspberry Pi libraries not found. Running in computer mode.")

class VideoStreamServer:
    def __init__(self, host='0.0.0.0', port=8080, is_pi=None):
        """
        Initialize the video streaming server
        
        Args:
            host (str): Host IP to bind to. Default '0.0.0.0' (all interfaces)
            port (int): Port to bind to. Default 8080
            is_pi (bool): Force Pi mode (True) or computer mode (False). None for auto-detect.
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.video_capture = None
        self.running = False
        self.px = None
        self.is_pi = IS_RASPBERRY_PI if is_pi is None else is_pi
        
        # Performance optimization settings
        self.frame_count = 0
        self.start_time = 0
        self.frame_interval = 0.03  # Target 30+ FPS (33ms between frames)
        self.jpeg_quality = 70  # Slightly lower quality for faster transmission (was 80)
        self.resolution = (640, 480)  # Increased resolution from 320x240 to 640x480
        self.send_buffer_size = 1000000  # 1MB buffer for socket operations
        self.monitor_interval = 30  # Print stats every 30 frames
        
        # Adaptive parameters
        self.min_frame_interval = 0.01  # Maximum 100 FPS
        self.max_frame_interval = 0.1   # Minimum 10 FPS
        self.target_packet_size = 40    # Increased target packet size for higher resolution
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
            
            # Initialize PiCar if on Raspberry Pi
            if self.is_pi:
                try:
                    self.px = Picarx()
                    print("PiCar-X initialized")
                except Exception as e:
                    print(f"Error initializing PiCar: {e}")
            else:
                print("Running in computer mode - PiCar functionality disabled")
                
            return True
        
        except Exception as e:
            print(f"Error setting up server: {e}")
            self.cleanup()
            return False
    
    def init_camera(self):
        """Initialize the camera based on platform"""
        if self.is_pi:
            # Initialize Picamera2 with optimized settings
            self.video_capture = Picamera2()
            config = self.video_capture.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            self.video_capture.configure(config)
            self.video_capture.start()
        else:
            # Initialize webcam using OpenCV
            self.video_capture = cv2.VideoCapture(0)
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Check if camera opened successfully
            if not self.video_capture.isOpened():
                print("Error: Could not open webcam")
                return False
                
        # Wait a moment for camera to initialize
        time.sleep(0.5)
        return True
    
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
                
                # Initialize camera
                if not self.init_camera():
                    print("Failed to initialize camera")
                    if self.client_socket:
                        self.client_socket.close()
                        self.client_socket = None
                    continue
                
                # Wait a moment for camera to initialize
                time.sleep(0.5)
                
                # Stream video to the client
                self.stream_to_client()
                
                # Clean up client resources
                if self.client_socket:
                    self.client_socket.close()
                    self.client_socket = None
                
                # Release camera
                self.release_camera()
        
        except KeyboardInterrupt:
            print("\nServer stopped by user")
        except Exception as e:
            print(f"Error in streaming: {e}")
        finally:
            self.cleanup()
    
    def release_camera(self):
        """Release camera resources based on platform"""
        if self.video_capture:
            if self.is_pi:
                if hasattr(self.video_capture, 'stop'):
                    self.video_capture.stop()
            else:
                self.video_capture.release()
            self.video_capture = None
    
    def get_frame(self):
        """Get a frame from the camera based on platform"""
        frame = None
        if self.is_pi:
            # PiCamera2 capture
            frame = self.video_capture.capture_array()
        else:
            # OpenCV capture
            ret, frame = self.video_capture.read()
            if not ret:
                print("Error: Failed to capture frame")
                return None
        
        return frame
            
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
                
                # Read a frame from the camera
                frame = self.get_frame()
                if frame is None:
                    continue
                
                # Get sensor data (less frequently than frames)
                if current_time - last_sensor_read_time > sensor_read_interval:
                    if self.is_pi and self.px:
                        try:
                            ultrasonic_data = self.px.ultrasonic.read()
                            last_sensor_read_time = current_time
                        except:
                            # If sensor read fails, just continue with last value
                            pass
                    else:
                        # Simulate ultrasonic data in computer mode
                        ultrasonic_data = 100.0  # Fixed distance of 100cm
                
                # Convert from BGR to RGB if using OpenCV
                if not self.is_pi and frame.shape[2] == 3:
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
        self.release_camera()
        
        print("Server resources cleaned up")
        
class ControlServer:
    def __init__(self, host='0.0.0.0', port=9090, is_pi=None):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = True
        self.current_steering = 0.0
        self.current_speed = 0.0
        self.client_connections = []
        self.is_pi = IS_RASPBERRY_PI if is_pi is None else is_pi
        self.picar = None
        self.car_controller = None
        
    def start(self):
        """Start the control server"""
        try:
            # Create socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind the socket
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            print(f"Control server started on {self.host}:{self.port}")
            
            if self.is_pi:
                self.car_controller = MovementController()
            
            # Start accepting connections
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    print(f"Control connection established from {client_address}")
                    
                    # Start a new thread to handle this client
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                    self.client_connections.append((client_socket, client_thread))
                    
                except Exception as e:
                    print(f"Error accepting connection: {e}")
                    
        except Exception as e:
            print(f"Control server error: {e}")
        finally:
            self.cleanup()
    
    def handle_client(self, client_socket, client_address):
        """Handle client connection and process commands"""
        buffer = b""
        
        try:
            while self.running:
                # Receive data
                data = client_socket.recv(1024)
                if not data:
                    break
                    
                buffer += data
                
                # Process complete messages (delimited by newlines)
                while b"\n" in buffer:
                    # Split at the first newline
                    line, buffer = buffer.split(b"\n", 1)
                    
                    try:
                        # Parse JSON message
                        message = json.loads(line.decode('utf-8'))
                        
                        # Check message type
                        if message.get("type") == "control_commands":
                            steering_angle = message.get("steering_angle", 0.0)
                            speed = message.get("speed", 0.0)
                            
                            # Update current values
                            self.current_steering = steering_angle
                            self.current_speed = speed
                            
                            # Apply control commands to the motor
                            self.apply_control_commands(steering_angle, speed)
                            
                            # Print for debugging
                            print(f"Received control: steering={steering_angle:.2f}, speed={speed:.2f}")
                            
                    except json.JSONDecodeError:
                        print(f"Invalid JSON message: {line}")
                    except Exception as e:
                        print(f"Error processing message: {e}")
        
        except Exception as e:
            print(f"Client handler error: {e}")
        finally:
            print(f"Client disconnected: {client_address}")
            client_socket.close()
            
    def apply_control_commands(self, steering_angle, speed):
        """
        Apply steering and speed commands to the robot's motors
        
        Args:
            steering_angle: Float between -1.0 and 1.0
            speed: Float between -1.0 and 1.0
        """
        if not self.is_pi:
            # Just print the commands in computer mode
            print(f"SIMULATION: Steering: {steering_angle:.2f}, Speed: {speed:.2f}")
            return
            
        # Implementation for Raspberry Pi with actual hardware
        try:
            if self.car_controller is not None:
                motor_speed = abs(speed * 100.0)
                
                # Have a threshold of 0.5
                if steering_angle < 0.5 and steering_angle > -0.5:
                    self.car_controller.move_forward(motor_speed)
                elif steering_angle < 0:
                    self.car_controller.turn_left(steering_angle, motor_speed)
                elif steering_angle > 0:
                    self.car_controller.turn_right(steering_angle, motor_speed)
                else:
                    self.car_controller.move_forward(motor_speed)
        except Exception as e:
            print(f"Error applying control commands: {e}")
            
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        # Close all client connections
        for client_socket, _ in self.client_connections:
            try:
                client_socket.close()
            except:
                pass
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            
        print("Control server cleaned up")

def main():
    # Ask if running on Raspberry Pi or computer
    try:
        mode = input("Running on Raspberry Pi (0) or Computer (1)? ")
        is_pi = mode.strip() != "1"
    except:
        # Default to auto-detect if input fails
        is_pi = IS_RASPBERRY_PI
    
    if is_pi:
        print("Running in Raspberry Pi mode with hardware support")
    else:
        print("Running in computer mode with simulation")
    
    # Create servers with selected mode
    video_server = VideoStreamServer(is_pi=is_pi)
    control_server = ControlServer(is_pi=is_pi)
    
    # Share PiCar instance if on Raspberry Pi
    if is_pi:
        try:
            picar = Picarx()
            video_server.px = picar
            control_server.picar = picar
            print("PiCar-X initialized and shared between servers")
        except Exception as e:
            print(f"Error initializing shared PiCar: {e}")
    
    # Start servers in separate threads
    video_thread = threading.Thread(target=video_server.start_streaming)
    video_thread.daemon = True
    video_thread.start()
    
    control_thread = threading.Thread(target=control_server.start)
    control_thread.daemon = True
    control_thread.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServers stopped by user")
    finally:
        video_server.cleanup()
        control_server.cleanup()

if __name__ == "__main__":
    main()

