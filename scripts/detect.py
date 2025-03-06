from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.5, iou=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Detection & Tracking", annotated_frame)

    # Para sair da janela, pressione 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
