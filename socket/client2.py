import socket
import cv2
import pickle
import struct

def main():
    # Replace with your server's IP
    server_ip = '10.12.81.99'  
    port = 9000

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, port))
    print(f"[+] Connected to server at {server_ip}:{port}")

    data = b""
    payload_size = struct.calcsize("L")  # L = unsigned long

    try:
        while True:
            # Receive message size
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet:
                    break
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]

            # Receive frame data
            while len(data) < msg_size:
                data += client_socket.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # Deserialize frame
            frame = pickle.loads(frame_data)

            # Display
            cv2.imshow("Client View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[*] Quit requested")
                break

    except KeyboardInterrupt:
        print("\n[*] Keyboard interrupt received. Exiting...")
    except Exception as e:
        print(f"[!] Error: {e}")
    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        print("[*] Disconnected from server.")

if __name__ == "__main__":
    main()
