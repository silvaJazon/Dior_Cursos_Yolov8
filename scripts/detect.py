from ultralytics import YOLO
import cv2

# Carregar o modelo pré-treinado
model = YOLO("yolov8n.pt")

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)  # Webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rodar o modelo de detecção e rastreamento
    results = model.track(frame, persist=True)  # "persist" mantém o rastreamento entre os quadros

    # Acessar o frame anotado (com caixas de detecção e rastreamento)
    annotated_frame = results[0].plot()

    # Mostrar a imagem com as detecções e rastreamento
    cv2.imshow("YOLOv8 Detection & Tracking", annotated_frame)

    # Para sair da janela, pressione 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
