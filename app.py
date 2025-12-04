from app_files.Security import SystemControl
import numpy as np
import cv2

video_path = input('Enter video path: ')
Analyzator = SystemControl('yolo11n.pt')

cap = cv2.VideoCapture('test.mp4')
writer = cv2.VideoWriter('detected_video.mp4', cv2.VideoWriter.fourcc(*'mp4v'), 20.0, (1024, 736))

while True:
    success, frame = cap.read()
    if not success: break
    frame = frame[:, 128:frame.shape[1]-128]
    frame = np.concatenate([np.zeros((8, 1024, 3), dtype=np.uint8), frame, np.zeros((8, 1024, 3), dtype=np.uint8)], axis=0)
    frame = Analyzator(frame)
    writer.write(frame)
cap.release()
writer.release()