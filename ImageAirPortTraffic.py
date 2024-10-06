from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

img_path = '/Users/arnavsaxena/Desktop/Coding/Projects/Airport_Traffic_Detection/Videos/Q4.webp'
img = cv2.imread(img_path)

if img is None:
    print("Failed to load image")
    
else:
    results = model(img)

    person_count = 0
    for result in results[0].boxes:
        if result.cls == 0: 
            person_count += 1

            x1, y1, x2, y2 = result.xyxy[0].int().tolist()

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'no:{person_count}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    
    traffic = 'HIGH' if person_count > 30 else 'LOW' if person_count < 15 else 'FINE'
    colour = (0, 0, 255) if person_count > 30 else (0, 255, 0) if person_count < 15 else (255, 0, 0)
    cv2.putText(img, traffic, (img.shape[1] - 200, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, colour, 5)

    total_label = f'Total: {person_count}'
    cv2.putText(img, total_label, (img.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('YOLOv8 Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()