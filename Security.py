from datetime import datetime, timedelta
from ultralytics import YOLO
import cv2
import os

class SystemControl:
    '''Инициализируем объект класса для обработки кадра на наличие людей в запрещенных зонах'''
    def __init__(self, security_zones, model_name, delay=3):
        self.border_zones = security_zones
        self.model = YOLO(f'models/{model_name}')
        self.alarm_flag = False
        self.alarm_timer = None
        self.alarm_delay = timedelta(seconds=delay)

    '''Метод для обработки кадра'''
    def __call__(self, frame):
        cv2.imwrite('frame.jpg', frame)
        result = self.model('frame.jpg', imgsz=frame.shape[:2])[0]
        os.remove('frame.jpg')
        frame = self.build_security_zone(frame)
        for box in result.boxes:
            if box.cls[0] != 0: continue
            frame = self.person_processing(frame, box.xyxy[0])
        return frame

    '''Метод для отрисовки запретных зон'''
    def build_security_zone(self, frame):
        cv2.polylines(frame, self.border_zones, isClosed=True, color=(10, 0, 255), thickness=2)
        return frame

    ''''''
    def person_processing(self, frame, box_coords):
        x1, y1, x2, y2 = box_coords
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        person_inside = any(self.person_in_zone([x1, y1, x2, y2], zone) for zone in self.border_zones)
        if person_inside:
            self.alarm_flag = True
            self.alarm_timer = datetime.now() + self.alarm_delay
        if self.alarm_flag:
            if datetime.now() <= self.alarm_timer:
                cv2.putText(frame, 'Alarm!', (frame.shape[1] // 4 + 50, frame.shape[0] // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 2)
            else:
                self.alarm_flag = False
                self.alarm_timer = None
        return frame

    '''Метод проверяет находится ли человек на кадре в запрещенной зоне'''
    @staticmethod
    def person_in_zone(xyxy, zone):
        x1, y1, x2, y2 = xyxy
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        intersects = cv2.pointPolygonTest(zone, (x, y), False)
        return intersects >= 0