# Im still learning this
# Solved the github issue hopefully ;)

from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

video_path = '/Users/arnavsaxena/Desktop/Coding/Projects/Airport_Traffic_Detection/Videos/V5.mp4'
cap = cv2.VideoCapture(video_path)

desired_fps = 1000
cap.set(cv2.CAP_PROP_FPS, desired_fps)
delay = int(1000 / desired_fps)

with open('output.txt', 'w') as file:
    file.write('0' + '\n')
    file.write('1')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)

    with open('output.txt', 'r') as file:
        file_content = file.read()

    values = file_content.split('\n')

    
    person_count = sum(1 for box in results[0].boxes if box.cls == 0)
    label = f'Total: {person_count}'
    cv2.putText(frame, label, (frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    range = (values[0] - (values[0]/4),values[0] + (values[0]/4))
    traffic, colour = ('HIGH', (0, 0, 255)) if person_count > range[1] else ('LOW', (255, 0, 0)) if person_count < range[0] else ('FINE', (0, 255, 0))
    cv2.putText(frame, traffic, (frame.shape[1] - 200, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, colour, 5)
    cv2.imshow('YOLOv8 Detection', frame)


    new_average = ((values[0] * values[1]) + person_count)/(values[1] + 1)
    with open('output.txt', 'w') as file:
        file.write(new_average + '\n')
        file.write(values[1] + 1)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#single person detection