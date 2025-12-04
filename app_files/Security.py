from datetime import datetime, timedelta
from ultralytics import YOLO
import numpy as np
import json
import cv2
import os

class SystemControl:
    '''Инициализируем объект класса для обработки кадра на наличие людей в запрещенных зонах'''
    def __init__(self, model_name, delay=3):
        with open('./app_files/restricted_zones.json', 'r') as file:
            data = json.load(file)
            self.__border_zones = [np.array(figure) for figure in data]
        self.__model = YOLO(f'./app_files/models/{model_name}')
        self.__alarm_flag = False
        self.__alarm_timer = None
        self.__alarm_delay = timedelta(seconds=delay)

    '''Метод для обработки кадра'''
    def __call__(self, frame):
        cv2.imwrite('frame.jpg', frame)
        result = self.__model('frame.jpg', imgsz=frame.shape[:2])[0]
        os.remove('frame.jpg')
        frame = self.__build_security_zone(frame)
        for box in result.boxes:
            if box.cls[0] != 0: continue
            frame = self.__person_processing(frame, box.xyxy[0])
        return frame

    '''Метод для отрисовки запретных зон'''
    def __build_security_zone(self, frame):
        cv2.polylines(frame, self.__border_zones, isClosed=True, color=(10, 0, 255), thickness=2)
        return frame

    ''''''
    def __person_processing(self, frame, box_coords):
        x1, y1, x2, y2 = box_coords
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        person_inside = any(self.__person_in_zone([x1, y1, x2, y2], zone) for zone in self.__border_zones)
        if person_inside:
            self.__alarm_flag = True
            self.__alarm_timer = datetime.now() + self.__alarm_delay
        if self.__alarm_flag:
            if datetime.now() <= self.__alarm_timer:
                cv2.putText(frame, 'Alarm!', (frame.shape[1] // 4 + 50, frame.shape[0] // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 2)
            else:
                self.__alarm_flag = False
                self.__alarm_timer = None
        return frame

    '''Метод проверяет находится ли человек на кадре в запрещенной зоне'''
    @staticmethod
    def __person_in_zone(xyxy, zone):
        x1, y1, x2, y2 = xyxy
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        intersects = cv2.pointPolygonTest(zone, (x, y), False)
        return intersects >= 0