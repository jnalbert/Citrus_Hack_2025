import socket
import time
import sys
import os
from pathlib import Path
import base64
import json
from PIL import Image
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
            print("Connected! You can start chatting or sending media.")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def send_media(self, file_path, media_type):
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return False

            # Read and encode the file
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
                encoded_data = base64.b64encode(file_bytes).decode('utf-8')

            # Create JSON payload
            payload = {
                'media_type': media_type,
                'data': encoded_data
            }
            
            # Convert to JSON string and encode
            json_data = json.dumps(payload).encode('utf-8')
            
            # Send data length first
            self.socket.sendall(str(len(json_data)).encode('utf-8'))
            
            # Wait for acknowledgment
            self.socket.recv(1024)
            
            # Send the JSON data
            self.socket.sendall(json_data)
            
            print(f"{media_type.capitalize()} {Path(file_path).name} sent successfully!")
            return True
            
        except Exception as e:
            print(f"Error sending {media_type}: {e}")
            return False

    def receive_media(self, data_size):
        try:
            # Create received_media directory if it doesn't exist
            if not os.path.exists('received_media'):
                os.makedirs('received_media')

            # Send acknowledgment
            self.socket.sendall(b"OK")

            # Receive the JSON data
            json_data = b""
            remaining = int(data_size)
            while remaining > 0:
                data = self.socket.recv(min(remaining, 4096))
                if not data:
                    break
                json_data += data
                remaining -= len(data)

            # Parse JSON
            payload = json.loads(json_data.decode('utf-8'))
            media_type = payload['media_type']
            encoded_data = payload['data']

            # Decode base64 data
            media_bytes = base64.b64decode(encoded_data)
            
            # Generate unique filename with appropriate extension
            ext = '.jpg' if media_type == 'image' else '.mp4'
            filename = f"received_media/{media_type}_{int(time.time())}{ext}"
            
            # Save the media file
            with open(filename, 'wb') as f:
                f.write(media_bytes)
            
            print(f"{media_type.capitalize()} received and saved as {filename}")
            return True
            
        except Exception as e:
            print(f"Error receiving media: {e}")
            return False

    def chat(self):
        if not self.socket:
            print("Not connected to friend")
            return False
        
        try:
            print("Commands: ")
            print("  'send image <path>' to send an image")
            print("  'send video <path>' to send a video")
            print("  'quit' to exit")
            
            # Start chat loop
            while self.running:
                # Get message to send
                message = input("You: ")
                if message.lower() == 'quit':
                    break
                
                # Check if it's a media send command
                if message.lower().startswith('send '):
                    parts = message.split(maxsplit=2)
                    if len(parts) == 3:
                        media_type = parts[1]
                        file_path = parts[2].strip()
                        if media_type in ['image', 'video']:
                            self.send_media(file_path, media_type)
                            continue
                
                # Send regular message as JSON
                payload = {
                    'media_type': 'text',
                    'data': message
                }
                json_data = json.dumps(payload).encode('utf-8')
                self.socket.sendall(str(len(json_data)).encode('utf-8'))
                self.socket.recv(1024)  # Wait for ack
                self.socket.sendall(json_data)
                
                # Get response from friend
                size = self.socket.recv(1024).decode('utf-8')
                if not size:
                    print("Friend disconnected")
                    break

                # Receive and process the response
                self.socket.sendall(b"OK")
                response_data = b""
                remaining = int(size)
                while remaining > 0:
                    data = self.socket.recv(min(remaining, 4096))
                    if not data:
                        break
                    response_data += data
                    remaining -= len(data)

                response = json.loads(response_data.decode('utf-8'))
                if response['media_type'] in ['image', 'video']:
                    self.receive_media(size)
                else:
                    print(f"Friend: {response['data']}")
                
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
