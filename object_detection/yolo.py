import cv2
import time
import logging
from ultralytics import YOLOE

# ─── SETUP LOGGING & FPS COUNTERS ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)
frame_count = 0
start_time = time.time()
prev_time = start_time

# ─── LOAD MODEL & CAMERA ─────────────────────────────────────────────────────
yolo = YOLOE('yoloe-11s-seg-pf.pt')
videoCap = cv2.VideoCapture(0)

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [
        (base_colors[color_index][i] + increments[color_index][i] * (cls_num // len(base_colors))) % 256
        for i in range(3)
    ]
    return tuple(color)

# Default color for unknown objects
unknown_color = (128, 128, 128)  # Grey color for unknown objects

# ─── INFERENCE LOOP ──────────────────────────────────────────────────────────
while True:
    ret, frame = videoCap.read()
    if not ret:
        continue

    # detect & track
    results = yolo.predict(frame, stream=True, conf=0.3)

    # draw boxes
    for result in results:
        classes_names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cls  = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < 0.4:
                class_name = "Unknown Object"
                colour     = unknown_color
            else:
                class_name = classes_names[cls]
                colour     = getColours(cls)

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            label = f'{class_name} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    # ── FPS CALC & OVERLAY ───────────────────────────────────────────────────
    frame_count += 1
    now = time.time()
    fps = 1.0 / (now - prev_time)
    prev_time = now

    # overlay instantaneous FPS
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # log average FPS every second
    if now - start_time >= 1.0:
        avg_fps = frame_count / (now - start_time)
        logging.info(f'Avg FPS: {avg_fps:.1f} over {frame_count} frames')
        start_time = now
        frame_count = 0

    # display
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
videoCap.release()
cv2.destroyAllWindows()