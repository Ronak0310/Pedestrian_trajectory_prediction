from pathlib import Path
from turtle import color
import cv2
import numpy as np
from collections import defaultdict
import os
from kalmanfilter import KalmanFilter

class Colors:
    def __init__(self):
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()

class Visualizer():
    def __init__(self, names):
        
        self.textColor = (255,255,255)
        self.names = names
        self.count = 0  # variable to update default_dict after certain number of count
        self.kf = KalmanFilter()
            
    def drawEmpty(self, frame, frameCount):
        """For images with no detections, displaying minimap and updating trajectory values
        
        Returns:
            frame (image): Image with minimap (if minimap enabled)
        """     

        return frame
    
    def drawBBOX(self, xyxy, frame, frameCount):
        """Draws just the BBOX with the class name and confidence score

        Args:
            xyxy (array): output from inference
            frame (image): Image to draw

        Returns:
            frame (image): Image with all the BBOXes
        """
        for detection in xyxy:  
            x1, y1, x2, y2 = detection[0:4]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            try:
                conf_score = round(detection[4].item() * 100, 1)
            except AttributeError:
                conf_score = round(detection[4] * 100, 1)

            try:
                classID = int(detection[5].item())
            except AttributeError:
                classID = int(detection[5])
                
            color = colors(classID)
            cls = self.names[classID] if self.names else classID
            
            # Displays the main bbox and add overlay to make bbox transparent
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Finds the space required for text
            textLabel = f'{cls} {conf_score}%'
            (w1, h1), _ = cv2.getTextSize(
                textLabel, 0, 0.3, 1
            )

            # Displays BG Box for the text and text itself
            cv2.rectangle(overlay, (x1, y1 - 10), (x1 + w1, y1), color, -1, cv2.LINE_AA)
            image = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            frame = cv2.putText(
                image, textLabel, (x1, y1 - 3), 
                0, 0.3, self.textColor, 1, cv2.LINE_AA
            )


        return frame

    def drawTracker(self, trackers, frame, frameCount):
        """Draws the BBOX along with Tracker ID for each BBOX

        Args:
            trackers (array): SORT Tracker object
            frame (image): Image to draw

        Returns:
            image: Image with tracker id and bbox
        """
        
        for detection in trackers:
            x1, y1, x2, y2 = detection[0:4]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            conf_score = round(detection[4] * 100, 1)
            classID = int(detection[5])
            tracker_id = int(detection[9])

            color = colors(classID)
            cls = self.names[classID] if self.names else classID

            cx1, cy1 = int(detection[-4]), int(detection[-3])   # previous frame points
            track_pts = detection[-2]
            box_pts = detection[-1]
            
            # Displays the main bbox and add overlay to make bbox transparent
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=1, lineType=cv2.LINE_AA)

            # Finds the space required for text
            TrackerLabel = f'Track ID: {tracker_id}'
            (w1, h1), _ = cv2.getTextSize(
                TrackerLabel, 0, fontScale=0.3, thickness=1
            )
            baseLabel = f'{cls} {conf_score}%'
            (w2, h2), _ = cv2.getTextSize(
                baseLabel, 0, fontScale=0.3, thickness=1
            )

            # Displays BG Box for the text and text itself
            cv2.rectangle(overlay, (x1, y1 - 20), (x1 + w1, y1), color, -1, cv2.LINE_AA)
            cv2.rectangle(overlay, (x1, y1 - 10), (x1 + w2, y1), color, -1, cv2.LINE_AA)
            image = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            frame = cv2.putText(
                image, TrackerLabel, (x1, y1 - 13), 
                0, 0.3, self.textColor, 1, cv2.LINE_AA
            )
            frame = cv2.putText(
                image, baseLabel, (x1, y1 - 3), 
                0, 0.3, self.textColor, 1, cv2.LINE_AA
            )
            
            # Use kalman_filter to predict next point and draw heading arrow in that direction
            if type(track_pts)==defaultdict:
                for pt in track_pts[tracker_id]:
                    # cv2.circle(frame, (pt[0], pt[1]), 1, (0,0,255), -1, cv2.LINE_AA)
                    predicted = self.kf.predict(pt[0], pt[1])
                pred = predicted
                for i in range(2):
                    pred = self.kf.predict(pred[0], pred[1])
                cv2.arrowedLine(frame, (cx1,cy1), (int(pred[0]),int(pred[1])), (255,0,0),1)
            
            # if type(box_pts)==defaultdict:
            #     for pt in box_pts[tracker_id]:
            #         predicted_box_left = self.kf.predict(pt[0], pt[1])
            #         predicted_box_right = self.kf.predict(pt[2], pt[3])
            #     # predicted_box_left = self.kf.predict(box_pts[tracker_id][-1][0], box_pts[tracker_id][-1][1])
            #     # predicted_box_right = self.kf.predict(box_pts[tracker_id][-1][2], box_pts[tracker_id][-1][3])
            #     pred_box_left = predicted_box_left
            #     pred_box_right = predicted_box_right
            #     # for i in range(2):
            #     #     pred_box_left = self.kf.predict(pred_box_left[0], pred_box_left[1])
            #     #     pred_box_right = self.kf.predict(pred_box_right[0], pred_box_right[1])
                
            #     cv2.rectangle(frame, (int(pred_box_left[0]), int(pred_box_left[1])), 
            #                  (int(pred_box_right[0]), int(pred_box_right[1])), 
            #                  (0,0,255), thickness=1, lineType=cv2.LINE_AA)


        return frame
