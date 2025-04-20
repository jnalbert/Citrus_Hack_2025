import socket
import sys
import threading
import time
import os
import base64
import json

class SocketClient:
    def __init__(self, host='10.12.81.101', port=5000):
        self.host = host
        self.port = port
        self.client_socket = None

    def connect(self):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            print(f"Connected to server at {self.host}:{self.port}")
            self._chat()
        except Exception as e:
            print(f"Connection failed: {e}")
            self.disconnect()

    def _chat(self):
        try:
            while True:
                user_input = input("Enter message or type 'send image <path>' or 'send video <path>': ")
                if user_input.lower() == 'quit':
                    break
                elif user_input.startswith('send image '):
                    self._send_media(user_input[11:], 'image')
                elif user_input.startswith('send video '):
                    self._send_media(user_input[11:], 'video')
                else:
                    payload = {
                        'media_type': 'text',
                        'data': user_input
                    }
                    self._send_json(payload)

                    # Wait for reply
                    self._receive_response()

        except Exception as e:
            print(f"Error during chat: {e}")
        finally:
            self.disconnect()

    def _send_json(self, payload):
        try:
            data = json.dumps(payload).encode('utf-8')
            self.client_socket.sendall(str(len(data)).encode('utf-8'))
            self.client_socket.recv(1024)  # wait for OK
            self.client_socket.sendall(data)
        except Exception as e:
            print(f"Error sending data: {e}")

    def _send_media(self, filepath, media_type):
        try:
            if not os.path.exists(filepath):
                print("File not found.")
                return

            with open(filepath, 'rb') as f:
                encoded_data = base64.b64encode(f.read()).decode('utf-8')

            payload = {
                'media_type': media_type,
                'data': encoded_data
            }
            print(payload);
            self._send_json(payload)
            print(f"{media_type.capitalize()} sent successfully.")
        except Exception as e:
            print(f"Error sending {media_type}: {e}")

    def _receive_response(self):
        try:
            data_size = self.client_socket.recv(1024).decode('utf-8')
            if not data_size:
                return

            self.client_socket.sendall(b"OK")

            json_data = b""
            remaining = int(data_size)
            while remaining > 0:
                data = self.client_socket.recv(min(remaining, 4096))
                if not data:
                    break
                json_data += data
                remaining -= len(data)

            message = json.loads(json_data.decode('utf-8'))
            print(f"Friend: {message['data']}")
        except Exception as e:
            print(f"Error receiving message: {e}")

    def disconnect(self):
        if self.client_socket:
            self.client_socket.close()
            print("Disconnected from server.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <server_ip>")
        sys.exit(1)

    server_ip = sys.argv[1]
    client = SocketClient(host=server_ip)
    client.connect()


if __name__ == '__main__':
    main()