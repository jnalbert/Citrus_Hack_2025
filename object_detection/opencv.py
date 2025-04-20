import cv2
videoCap = cv2.VideoCapture(0)

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue

    sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, mask = sal.computeSaliency(frame)
    mask = (mask*255).astype('uint8')
    _, binmask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c)>500]

    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Image', frame)

     # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release the video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()

