import cv2
from ultralytics import YOLO

# Load the model
yolo = YOLO('yolov8s.pt')

# Load the video capture
videoCap = cv2.VideoCapture(0)

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

# Default color for unknown objects
unknown_color = (128, 128, 128)  # Grey color for unknown objects

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue
    
    # Set conf parameter to 0 to detect all objects regardless of confidence
    results = yolo.track(frame, stream=True, conf=0.0)

    for result in results:
        # get the classes names
        classes_names = result.names

        # iterate over each box
        for box in result.boxes:
            # get coordinates
            [x1, y1, x2, y2] = box.xyxy[0]
            # convert to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # get the class
            cls = int(box.cls[0])
            
            # get the confidence
            conf = float(box.conf[0])

            # Determine if it's a known or unknown object based on confidence
            if conf < 0.4:
                # This is an unknown object
                class_name = "Unknown Object"
                colour = unknown_color
            else:
                # This is a classified object
                class_name = classes_names[cls]
                colour = getColours(cls)

            # draw the rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # put the class name and confidence on the image
            label = f'{class_name} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                
    # show the image
    cv2.imshow('frame', frame)

    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()



