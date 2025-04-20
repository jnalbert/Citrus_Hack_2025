import cv2
import time
import logging
from ultralytics import YOLOE

# ─── SETUP LOGGING ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)

# ─── 1) ONE‑TIME MODEL & CAMERA SETUP ────────────────────────────────────────
yolo = YOLOE('yoloe-11s-seg-pf.pt').to('mps')  # or 'cpu'

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ─── 2) INIT FPS COUNTER ─────────────────────────────────────────────────────
frame_count = 0
start_time = time.time()
prev_time = start_time

# ─── 3) INFERENCE LOOP ───────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run class‑agnostic predict
    results = yolo.predict(
        source        = frame,
        conf          = 0.2,
        iou           = 0.45,
        single_cls    = True,
        agnostic_nms  = True,
        max_det       = 50
    )
    res = results[0]

    # draw boxes
    ih, iw = res.orig_shape
    h_ratio = frame.shape[0] / ih
    w_ratio = frame.shape[1] / iw
    for box in res.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        x1, y1 = int(x1 * w_ratio), int(y1 * h_ratio)
        x2, y2 = int(x2 * w_ratio), int(y2 * h_ratio)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ── FPS CALC ───────────────────────────────────────────────────────────────
    frame_count += 1
    now = time.time()
    fps = 1.0 / (now - prev_time)
    prev_time = now

    # overlay FPS on the frame
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # log average FPS every second
    if now - start_time >= 1.0:
        avg_fps = frame_count / (now - start_time)
        logging.info(f'Avg FPS: {avg_fps:.1f} over {frame_count} frames')
        # reset counters
        start_time = now
        frame_count = 0

    # display
    cv2.imshow('objects', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()