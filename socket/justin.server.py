import socket
import pickle
import struct
import sys
import numpy as np
import time

# Use vilib instead of cv2
import vilib

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
        self.running = False
        self.display_id = None
    
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
                
                # Initialize vilib for camera access
                try:
                    print("Initializing vilib camera...")
                    vilib.init()
                    
                    # Start the camera with vilib
                    # Parameters may need adjustment based on your Pi camera
                    vilib.camera.start_camera(width=640, height=480, fps=30)
                    
                    # Initialize a display window if needed
                    self.display_id = vilib.display.add_display(width=640, height=480)
                    print("Camera initialized successfully")
                    
                    # Stream video to the client
                    self.stream_to_client()
                    
                except Exception as e:
                    print(f"Error initializing camera: {e}")
                    if self.client_socket:
                        self.client_socket.close()
                    continue
                
                # Clean up client resources
                if self.client_socket:
                    self.client_socket.close()
                    self.client_socket = None
                
                # Stop the camera
                try:
                    vilib.camera.stop_camera()
                except:
                    pass
                
                # Remove the display
                if self.display_id is not None:
                    try:
                        vilib.display.remove_display(self.display_id)
                        self.display_id = None
                    except:
                        pass
        
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
                # Get a frame from vilib camera
                try:
                    # Get the frame from vilib camera
                    frame = vilib.camera.get_frame()
                    
                    if frame is None or frame.size == 0:
                        print("Error: Failed to capture video frame")
                        time.sleep(0.1)  # Small delay before retrying
                        continue
                    
                    # Update the display if needed
                    if self.display_id is not None:
                        vilib.display.update_display(self.display_id, frame)
                    
                    # Serialize the frame using pickle
                    serialized_frame = pickle.dumps(frame)
                    
                    # Create message with frame size and frame data
                    message = struct.pack("L", len(serialized_frame)) + serialized_frame
                    
                    # Send the frame to the client
                    self.client_socket.sendall(message)
                    
                    # Brief delay to control frame rate if needed
                    # Adjust this value based on your requirements
                    time.sleep(0.03)  # ~30 FPS
                    
                except Exception as e:
                    print(f"Error capturing or sending frame: {e}")
                    time.sleep(0.1)  # Small delay before retrying
                
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
        
        # Clean up vilib resources
        try:
            if self.display_id is not None:
                vilib.display.remove_display(self.display_id)
            vilib.camera.stop_camera()
            vilib.exit()
        except Exception as e:
            print(f"Error cleaning up vilib: {e}")
        
        print("Server resources cleaned up")

def main():
    # Create server with default host and port
    server = VideoStreamServer()
    
    # Start streaming
    server.start_streaming()

if __name__ == "__main__":
    main()

