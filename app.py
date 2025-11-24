from datetime import datetime
from ultralytics import YOLO
import numpy as np
import cv2
import os

alarm_flag = False

def build_security_zone(frame, border_zones):
    cv2.polylines(frame, border_zones, isClosed=True, color=(5, 0, 255), thickness=2)
    return frame

def person_in_zone(xyxy, zone):
    x1, y1, x2, y2 = xyxy
    x, y = (x1 + x2) / 2, (y1 + y2) / 2
    intersects = cv2.pointPolygonTest(zone, (x, y), False)
    return intersects

def person_processing(frame, box_coords, border_zones):
    global alarm_flag
    timer = 0
    x1, y1, x2, y2 = box_coords
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for zone in border_zones:
        if person_in_zone([x1, y1, x2, y2], zone):
            alarm_flag = True
            timer = datetime.now()
        if alarm_flag:
            cv2.putText(frame, 'Alarm!!!', (frame.shape[1] // 2 - 20, frame.shape[0] // 2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 2)
            break
    if (datetime.now() - timer).total_seconds() / 60 >= 3: alarm_flag = False
    return frame

video_path = 'test.mp4'
frame_folder = 'frames'
model_folder = 'models'

coords = []
with open('security_zone_coords.txt', 'r') as file:
    for line in file.readlines():
        x, y = line.split(' ')
        x, y = int(x), int(y)
        coords.append([x, y])
security_coords = [np.array(coords[i:i+4]) for i in range(0, len(coords), 4)]

model = YOLO(f'{model_folder}/yolo11n.pt')
cap = cv2.VideoCapture(video_path)
writer = cv2.VideoWriter('detected_video.avi', cv2.VideoWriter.fourcc(*'XVID'), 20.0, (720, 1280))
if not os.path.exists(frame_folder):
    os.mkdir(frame_folder)

while True:
    success, frame = cap.read()
    if not success: break
    frame = frame[:, 128:frame.shape[1]-128]
    frame = np.concatenate([np.zeros((8, 1024, 3), dtype=np.uint8), frame, np.zeros((8, 1024, 3), dtype=np.uint8)], axis=0)
    cv2.imwrite('frame.jpg', frame)
    result = model('frame.jpg', imgsz=(736, 1024))[0]
    os.remove('frame.jpg')
    frame = build_security_zone(frame, security_coords)
    for box in result.boxes:
        if box.cls[0] != 0: continue
        frame = person_processing(frame, box.xyxy[0], security_coords)
    writer.write(frame)
cap.release()
writer.release()