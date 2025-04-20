import socket
import time
import json

class NavigationServer:
    def __init__(self, host='0.0.0.0', port=9090):
        self.host = host
        self.port = port
        self.client_conn = None
        self.client_addr = None
        self.server_socket = None

    def start_server(self):
        print(f"[SERVER] Starting navigation server on port {self.port}...")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"[SERVER] Waiting for client to connect on port {self.port}...")
        self.client_conn, self.client_addr = self.server_socket.accept()
        print(f"[SERVER] Client connected from {self.client_addr}")

        self.run_navigation_loop()

    def run_navigation_loop(self):
        try:
            while True:
                action = self.analyze_histogram_and_generate_action()
                self.send_action_to_client(action)
                time.sleep(1)  # simulate periodic updates
        except (ConnectionResetError, BrokenPipeError):
            print("[SERVER] Client disconnected")
        except Exception as e:
            print(f"[SERVER] Error: {e}")
        finally:
            self.cleanup()

    def analyze_histogram_and_generate_action(self):
        # Replace this with your histogram logic
        # Just a sample random behavior for now
        action = {
            "steering": 0.2,  # example value between -1 and 1
            "speed": 0.8      # example value between 0 and 1
        }
        print(f"[SERVER] Generated action: {action}")
        return action

    def send_action_to_client(self, action):
        if self.client_conn:
            message = json.dumps(action).encode('utf-8')
            self.client_conn.sendall(message)
            print(f"[SERVER] Sent action to client")

    def cleanup(self):
        if self.client_conn:
            self.client_conn.close()
        if self.server_socket:
            self.server_socket.close()
        print("[SERVER] Server shut down")

def main():
    server = NavigationServer()
    server.start_server()

if __name__ == "__main__":
    main()
