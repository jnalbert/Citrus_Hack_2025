import socket
import time
import sys
import os
from pathlib import Path
import base64
try:
    from PIL import Image
except ImportError:
    print("Please install Pillow with: pip install Pillow")
    sys.exit(1)
import io

class SocketClient:
    def __init__(self, host='10.12.81.101', port=5000):
        """Initialize client to connect to server
        Change host to friend's IP address and port to match server"""
        self.host = host
        self.port = port
        self.socket = None
        self.running = True

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f"Connecting to friend at {self.host}:{self.port}")
            self.socket.connect((self.host, self.port))
            print("Connected! You can start chatting or sending images.")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def send_image(self, image_path):
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                return False

            # Open and encode the image
            with Image.open(image_path) as img:
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format)
                img_byte_arr = img_byte_arr.getvalue()
                
                # Encode image bytes to base64
                encoded_img = base64.b64encode(img_byte_arr)
            
            # Send indicator that this is an encoded image
            self.socket.sendall(b"ENCODED_IMAGE:")
            
            # Send encoded image size
            self.socket.sendall(str(len(encoded_img)).encode('utf-8'))
            
            # Wait for acknowledgment
            self.socket.recv(1024)
            
            # Send the encoded image data
            self.socket.sendall(encoded_img)
            
            print(f"Image {Path(image_path).name} sent successfully!")
            return True
            
        except Exception as e:
            print(f"Error sending image: {e}")
            return False

    def receive_image(self, encoded_size):
        try:
            # Create received_images directory if it doesn't exist
            if not os.path.exists('received_images'):
                os.makedirs('received_images')

            # Send acknowledgment
            self.socket.sendall(b"OK")

            # Receive the encoded image data
            encoded_img = b""
            remaining = int(encoded_size)
            while remaining > 0:
                data = self.socket.recv(min(remaining, 4096))
                if not data:
                    break
                encoded_img += data
                remaining -= len(data)

            # Decode base64 image
            img_bytes = base64.b64decode(encoded_img)
            
            # Generate unique filename
            filename = f"received_images/image_{int(time.time())}.jpg"
            
            # Save the image
            with open(filename, 'wb') as f:
                f.write(img_bytes)
            
            print(f"Image received and saved as {filename}")
            return True
            
        except Exception as e:
            print(f"Error receiving image: {e}")
            return False

    def chat(self):
        if not self.socket:
            print("Not connected to friend")
            return False
        
        try:
            print("Commands: ")
            print("  'send <image_path>' to send an image")
            print("  'quit' to exit")
            
            # Start chat loop
            while self.running:
                # Get message to send
                message = input("You: ")
                if message.lower() == 'quit':
                    break
                
                # Check if it's an image send command
                if message.lower().startswith('send '):
                    image_path = message[5:].strip()
                    self.send_image(image_path)
                    continue
                
                # Send regular message
                self.socket.sendall(message.encode('utf-8'))
                
                # Get response from friend
                response = self.socket.recv(1024)
                if not response:
                    print("Friend disconnected")
                    break

                # Check if response is an encoded image
                response_str = response.decode('utf-8')
                if response_str.startswith("ENCODED_IMAGE:"):
                    # Get image size
                    size = self.socket.recv(1024).decode('utf-8')
                    self.receive_image(size)
                    continue

                print(f"Friend: {response_str}")
                
            return True
        except Exception as e:
            print(f"Chat error: {e}")
            return False

    def close(self):
        self.running = False
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except:
                pass
            self.socket = None
            print("Disconnected from friend")

def main():
    # Connect directly to friend's IP
    client = SocketClient()
    
    if client.connect():
        try:
            client.chat()
        except KeyboardInterrupt:
            print("\nEnding chat session...")
        finally:
            client.close()
            sys.exit(0)

if __name__ == "__main__":
    main()
