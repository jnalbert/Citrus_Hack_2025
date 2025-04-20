import os
import sys
import time
import threading
import cv2
from ultralytics import YOLO

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_menu():
    """Display the main menu."""
    clear_screen()
    print("==================================")
    print("       Computer Vision App        ")
    print("==================================")
    print("[l] - Listen to socket server")
    print("[q] - Quit")
    print("==================================")
    print("Enter your choice: ", end="", flush=True)

def socket_listener():
    """Start the socket server and listen for incoming connections."""
    from socket.server import SocketServer
    
    # Create and start the socket server
    server = SocketServer()
    try:
        server.start()
        
        print("\nSocket server is now listening...")
        print("Press Ctrl+C to stop the server.")
        
        # Keep the server running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping socket server...")
        server.stop()
    finally:
        print("Socket server stopped.")

def process_frames(frame):
    """Process video frames with object detection."""
    # Load the YOLO model
    yolo = YOLO('object_detection/yolov8n.pt')
    
    # Process the frame with YOLO
    results = yolo.predict(frame, conf=0.3)
    
    # Draw bounding boxes on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Get class name
            class_name = result.names[cls]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    """Main entry point for the application."""
    while True:
        display_menu()
        choice = input().lower()
        
        if choice == 'l':
            # Start socket server in a separate thread
            print("\nStarting socket server...")
            server_thread = threading.Thread(target=socket_listener)
            server_thread.daemon = True
            server_thread.start()
            
            input("\nPress Enter to return to the main menu...")
            
        elif choice == 'q':
            print("\nExiting application...")
            sys.exit(0)
            
        else:
            print("\nInvalid choice. Press Enter to continue...")
            input()

if __name__ == "__main__":
    main()
