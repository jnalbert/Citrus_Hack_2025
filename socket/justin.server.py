import socket
import cv2
import pickle
import struct
import sys
from picamera2 import Picamera2
from picarx import Picarx

class VideoStreamServer:
    def __init__(self, host='0.0.0.0', port=8080):
        """
        Initialize the video streaming server
        
        Args:
            host (str): Host IP to bind to. Default '0.0.0.0' (all interfaces)
            port (int): Port to bind to. Default 9999
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.video_capture = None
        self.running = False
        self.px = None
    
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
                
                # Initialize Picamera2
                self.video_capture = Picamera2()
                self.video_capture.preview_configuration.main.size = (640, 480)
                self.video_capture.preview_configuration.main.format = "RGB888"
                self.video_capture.configure("preview")
                self.video_capture.start()
                
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
    
    def stream_to_client(self):
        """Stream video frames to the connected client"""
        try:
            while self.running and self.client_socket:
                # Read a frame from the Picamera2
                frame = self.video_capture.capture_array()
                
                # Get sensor data
                ultrasonic_data = self.px.ultrasonic.read()
                print(f"Ultrasonic data: {ultrasonic_data}")
                
                # Create a data package containing both frame and sensor data
                data_package = {
                    'frame': frame,
                    'ultrasonic': ultrasonic_data,
                    # Add other sensor data here as needed
                    'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
                }
                
                # Convert from BGR to RGB if needed (depends on camera output)
                data_package['frame'] = cv2.cvtColor(data_package['frame'], cv2.COLOR_BGR2RGB)
                
                # Serialize the data package using pickle
                serialized_data = pickle.dumps(data_package)
                
                # Create message with package size and package data
                message = struct.pack("L", len(serialized_data)) + serialized_data
                
                # Send the data package to the client
                self.client_socket.sendall(message)
              
                
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

