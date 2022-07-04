from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
import os
from kalmanfilter import KalmanFilter

 
class Visualizer():
    def __init__(self):
        self.classID_dict = {
            0: ("Person", (0, 90, 255)), 
            1: ("Bicycle", (255, 90, 0)), 
            2: ("Car", (90, 255, 0)),
            3: ("Motorcycle", (204, 0, 102)),
            5: ("Bus", (0, 0, 255)),
            16: ("Dog", (0, 102, 204))
        }

        self.textColor = (0,0,0)
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
                
            color = self.classID_dict[classID][1]
            
            # Displays the main bbox and add overlay to make bbox transparent
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Finds the space required for text
            textLabel = f'{self.classID_dict[classID][0]} {conf_score}%'
            (w1, h1), _ = cv2.getTextSize(
                textLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )

            # Displays BG Box for the text and text itself
            cv2.rectangle(overlay, (x1, y1 - 20), (x1 + w1, y1), color, -1, cv2.LINE_AA)
            image = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            frame = cv2.putText(
                image, textLabel, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
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

            color = self.classID_dict[classID][1]

            cx1, cy1 = int(detection[-3]), int(detection[-2])   # previous frame points
            track_pts = detection[-1]
            
            # Displays the main bbox and add overlay to make bbox transparent
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Finds the space required for text
            TrackerLabel = f'Track ID: {tracker_id}'
            (w1, h1), _ = cv2.getTextSize(
                TrackerLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            baseLabel = f'{self.classID_dict[classID][0]} {conf_score}%'
            (w2, h2), _ = cv2.getTextSize(
                baseLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )

            # Displays BG Box for the text and text itself
            cv2.rectangle(overlay, (x1, y1 - 40), (x1 + w1, y1), color, -1, cv2.LINE_AA)
            cv2.rectangle(overlay, (x1, y1 - 20), (x1 + w2, y1), color, -1, cv2.LINE_AA)
            image = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            frame = cv2.putText(
                image, TrackerLabel, (x1, y1 - 24), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
            )
            frame = cv2.putText(
                image, baseLabel, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA
            )
            
            # Use kalman_filter to predict next point and draw heading arrow in that direction
            if type(track_pts)==defaultdict:
                for pt in track_pts[tracker_id]:
                    cv2.circle(frame, (pt[0], pt[1]), 1, (0,0,255), -1, cv2.LINE_AA)
                    predicted = self.kf.predict(pt[0], pt[1])
                pred = self.kf.predict(predicted[0], predicted[1])
                pred2 = self.kf.predict(pred[0], pred[1])
                cv2.arrowedLine(frame, (cx1,cy1), (int(pred2[0]),int(pred2[1])), (255,0,0),1)

        return frame
