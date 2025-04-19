import socket
import sys
import threading
import time
import os
import base64

class SocketServer:
    def __init__(self, host='0.0.0.0', port=5000):
        """Initialize server to accept connections from any IP (0.0.0.0)
        Change port if needed to match friend's client port"""
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.clients = []

    def start(self):
        try:
            # Create a TCP/IP socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Allow reuse of the address
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind the socket to the port
            print(f"Starting server on {self.host}:{self.port}")
            self.server_socket.bind((self.host, self.port))
            
            # Listen for incoming connections
            self.server_socket.listen(5)
            self.running = True
            
            # Print local IP for friend to connect to
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            print(f"Tell your friend to connect to: {local_ip}:{self.port}")
            print("Server is listening for connections...")
            
            # Accept connections in a separate thread
            accept_thread = threading.Thread(target=self._accept_connections)
            accept_thread.daemon = True
            accept_thread.start()
            
        except Exception as e:
            print(f"Error starting server: {e}")
            self.stop()

    def _accept_connections(self):
        while self.running:
            try:
                # Wait for a connection
                client_socket, client_address = self.server_socket.accept()
                print(f"Friend connected from {client_address}")
                
                # Add client to list
                self.clients.append(client_socket)
                
                # Handle client in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_address)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"Error accepting connection: {e}")

    def _receive_image(self, client_socket, encoded_size):
        try:
            # Create received_images directory if it doesn't exist
            if not os.path.exists('received_images'):
                os.makedirs('received_images')

            # Send acknowledgment
            client_socket.sendall(b"OK")

            # Receive the encoded image data
            encoded_img = b""
            remaining = int(encoded_size)
            while remaining > 0:
                data = client_socket.recv(min(remaining, 4096))
                if not data:
                    break
                encoded_img += data
                remaining -= len(data)

            print("Image data received successfully!")

            # Decode base64 image
            img_bytes = base64.b64decode(encoded_img)
            
            # Generate unique filename
            filename = f"received_images/image_{int(time.time())}.jpg"
            
            # Save the image
            with open(filename, 'wb') as f:
                f.write(img_bytes)
            
            print(f"Image saved successfully as {filename}")
            return True
            
        except Exception as e:
            print(f"Error receiving image: {e}")
            return False

    def _handle_client(self, client_socket, client_address):
        try:
            print(f"Start chatting with your friend at {client_address}!")
            while self.running:
                # Receive data from client
                data = client_socket.recv(1024)
                if not data:
                    break
                
                # Check if it's an image
                message = data.decode('utf-8')
                if message.startswith("ENCODED_IMAGE:"):
                    # Get the size of encoded image
                    encoded_size = client_socket.recv(1024).decode('utf-8')
                    self._receive_image(client_socket, encoded_size)
                    continue
                
                # Process received text message
                print(f"Friend: {message}")
                
                # Get your response and send it
                response = input("You: ")
                if response.lower() == 'quit':
                    break
                client_socket.sendall(response.encode('utf-8'))
                
        except Exception as e:
            print(f"Error chatting with friend {client_address}: {e}")
        finally:
            # Clean up the connection
            if client_socket in self.clients:
                self.clients.remove(client_socket)
            client_socket.close()
            print(f"Chat ended with friend at {client_address}")

    def stop(self):
        self.running = False
        if self.server_socket:
            # Close all client connections
            for client in self.clients:
                try:
                    client.close()
                except:
                    pass
            self.clients.clear()
            
            # Close server socket
            self.server_socket.close()
            print("Server stopped")

def main():
    # Create server that listens on all interfaces
    server = SocketServer()
    try:
        server.start()
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nEnding chat session...")
        server.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()