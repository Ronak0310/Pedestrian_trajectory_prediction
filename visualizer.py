from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
import os
import pandas as pd
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

class Minimap():
    def __init__(self, minimap_type='Terrain', minimap_coords=((361, 0), (638, 190)), trajectory_update_rate=30, trajectory_retain_duration=250):
        self.homography_CameraToMap = [[-7.69726646e-01, -3.92284300e+00,  5.63221161e+02],
                                      [-2.52034888e-01, -7.34562016e+00,  1.38008988e+03],
                                      [-4.02346894e-04, -8.87086124e-03, 1.00000000e+00]]
       
        if minimap_type == 'Terrain':
            self.Minimap = cv2.imread('./images/img2.png')
        elif minimap_type == 'Road':
            self.Minimap = cv2.imread('./map_files/map_cropped.png')
        else:
            print("Wrong Minimap type...defaulting to 'Terrain'")
            self.Minimap = cv2.imread('./images/img2.png')
        
        # Location in the main image to insert minimap
        self.locationMinimap = minimap_coords
        original_width = self.Minimap.shape[1]
        original_height = self.Minimap.shape[0]
 
        # Resizing the minimap accordingly
        resize_width = self.locationMinimap[1][0] - self.locationMinimap[0][0]
        resize_height = self.locationMinimap[1][1] - self.locationMinimap[0][1]

        self.Minimap = cv2.resize(self.Minimap, (resize_width, resize_height))

        self.width_scaling = resize_width/original_width
        self.height_scaling = resize_height/original_height

        self.realtime_trajectory = defaultdict(list)
        self.updateRate = trajectory_update_rate
        self.trajectory_retain_duration = trajectory_retain_duration

    def projection_image_to_map(self, x, y):
        """Converts image coordinates to minimap coordinates using loaded the homography matrix

        Returns:
          (int, int): x, y coordinates with respective to scaled minimap
        """
        pt1 = np.array([x, y, 1])
        pt1 = pt1.reshape(3, 1)
        pt2 = np.dot(self.homography_CameraToMap, pt1)
        pt2 = pt2 / pt2[2]
        return (int(pt2[0]*self.width_scaling), int(pt2[1]*self.height_scaling))

    def projection_image_to_map_noScaling(self, x, y):
        """Converts image coordinates to minimap coordinates using loaded the homography matrix

        Returns:
          (int, int): x, y coordinates with respective to scaled minimap
        """
        pt1 = np.array([x, y, 1])
        pt1 = pt1.reshape(3, 1)
        pt2 = np.dot(self.homography_CameraToMap, pt1)
        pt2 = pt2 / pt2[2]
        return (int(pt2[0]), int(pt2[1]))

    def update_realtime_trajectory(self, current_frameNumber):
        """Responsible for deleting trajectory points for each tracker id after 'self.trajectory_retain_duration' frames

        Returns:
            None
        """
        if self.realtime_trajectory:
            for keys, values in list(self.realtime_trajectory.items()):
                if len(values) == 0:
                    del self.realtime_trajectory[keys]
                elif current_frameNumber - values[0][3] > self.trajectory_retain_duration:
                    del self.realtime_trajectory[keys][0]



class Visualizer():
    def __init__(self, names):
        
        self.textColor = (255,255,255)
        self.names = names
        self.count = 0  # variable to update default_dict after certain number of count
        self.kf = KalmanFilter()
        self.showMinimap = False
        self.Minimap_obj = Minimap(trajectory_update_rate=5, trajectory_retain_duration=250)
        self.showTrajectory = False
        self.traj_dict = defaultdict(list)
    
    def draw_realtime_trajectory(self, minimap_img):
        """Displays the recorded trajectory onto the minimap

        Returns:
            None
        """
        if self.Minimap_obj.realtime_trajectory:
            for keys, values in self.Minimap_obj.realtime_trajectory.items():
                for v in values:
                    # color = self.classID_dict[v[2]][1]
                    color = colors(v[2])
                    cv2.circle(minimap_img, (int(v[0]),int(v[1])), 1, color, -1, cv2.LINE_AA)
        return minimap_img 
    
    def drawEmpty(self, frame, frameCount):
        """For images with no detections, displaying minimap and updating trajectory values
        
        Returns:
            frame (image): Image with minimap (if minimap enabled)
        """     
        if self.showMinimap:
            minimap_img = self.Minimap_obj.Minimap.copy()
            if self.showTrajectory:
                minimap_img = self.draw_realtime_trajectory(minimap_img)
                self.Minimap_obj.update_realtime_trajectory(frameCount)
            frame[self.Minimap_obj.locationMinimap[0][1]:self.Minimap_obj.locationMinimap[1][1], self.Minimap_obj.locationMinimap[0][0]:self.Minimap_obj.locationMinimap[1][0]] = minimap_img
            return frame
        else:
            return frame
    
    def drawBBOX(self, xyxy, frame, frameCount):
        """Draws just the BBOX with the class name and confidence score

        Args:
            xyxy (array): output from inference
            frame (image): Image to draw

        Returns:
            frame (image): Image with all the BBOXes
        """
        if self.showMinimap:
            minimap_img = self.Minimap_obj.Minimap.copy()
            if self.showTrajectory:
                minimap_img = self.draw_realtime_trajectory(minimap_img)
                self.Minimap_obj.update_realtime_trajectory(frameCount)
        
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

            if self.showMinimap:
                # Converting coordinates from image to map
                # Just using the larger y value because BBOX center is not were the foot/wheels of the classes are. So center point taken is the center of the bottom line of BBOX
                _, max_y = sorted((y1, y2))
                point_coordinates = self.Minimap_obj.projection_image_to_map((x1+x2)/2, max_y)
                cv2.circle(minimap_img, point_coordinates, 1, color, -1, cv2.LINE_AA)

        if self.showMinimap:      
            frame[self.Minimap_obj.locationMinimap[0][1]:self.Minimap_obj.locationMinimap[1][1], self.Minimap_obj.locationMinimap[0][0]:self.Minimap_obj.locationMinimap[1][0]] = minimap_img


        return frame

    def drawTracker(self, trackers, frame, frameCount):
        """Draws the BBOX along with Tracker ID for each BBOX

        Args:
            trackers (array): SORT Tracker object
            frame (image): Image to draw

        Returns:
            image: Image with tracker id and bbox
        """
        
        if self.showMinimap:
            minimap_img = self.Minimap_obj.Minimap.copy()
            if self.showTrajectory:
                minimap_img = self.draw_realtime_trajectory(minimap_img)
                self.Minimap_obj.update_realtime_trajectory(frameCount)
        
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

            cx1, cy1 = int(detection[10]), int(detection[11])   # previous frame points
            track_pts = detection[-1]

            # try approach for long term prediction
            center_x, center_y = int((x1 + x2)/2), int((y1+y2)/2)
            if frameCount % 3 ==0:
                self.traj_dict[tracker_id].append((center_x,center_y))
            
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
            
            # # Use 2D kalman_filter to predict Future points in the moving direction
            # # if type(track_pts)==defaultdict:
            # for pt in self.traj_dict[tracker_id]:
            #     cv2.circle(frame, (pt[0], pt[1]), 1, (0,0,255), -1, cv2.LINE_AA)
            # if len(self.traj_dict[tracker_id])>10:
            #     del self.traj_dict[tracker_id][0]
            #     # First without moving avraging
            #     # for pt in track_pts[tracker_id]:
            #     for pt in self.traj_dict[tracker_id]:
            #         # cv2.circle(frame, (pt[0], pt[1]), 1, (0,0,255), -1, cv2.LINE_AA)
            #         predicted = self.kf.predict(pt[0], pt[1])
            #         # w = pt[2]
            #         # h = pt[3]

            #     pred = predicted
            #     for i in range(10):
            #         pred = self.kf.predict(pred[0], pred[1])
            #         cv2.circle(frame, (int(pred[0]),int(pred[1])), 1, (255,0,0), -1, cv2.LINE_AA)
            #     # pred_box_left = (int(pred[0]-w/2) , int(pred[1]-h/2))
            #     # pred_box_right = (int(pred[0]+w/2) , int(pred[1]+h/2))
            # #     cv2.rectangle(frame, (int(pred_box_left[0]), int(pred_box_left[1])), 
            # #                  (int(pred_box_right[0]), int(pred_box_right[1])), 
            # #                  (0,0,255), thickness=1, lineType=cv2.LINE_AA)

                # Second with moving avaraging
                # df = pd.DataFrame.from_dict(track_pts[tracker_id])
                # df[2] = df[0].rolling(3).mean()
                # df[3] = df[1].rolling(3).mean()
                # df.dropna(inplace=True)
                # df = df.reset_index(drop=True)
                # x_y = (np.concatenate((np.array([df[2].tolist()]).T, np.array([df[3].tolist()]).T), axis=1))
                # for pt in x_y:
                #     cv2.circle(frame, (int(pt[0]),int(pt[1])), 1, (0,0,255), -1, cv2.LINE_AA)
                #     predicted = self.kf.predict(pt[0], pt[1])

                # pred = predicted
                # for i in range(10):
                #     pred = self.kf.predict(pred[0], pred[1])
                #     cv2.circle(frame, (int(pred[0]),int(pred[1])), 1, (255,0,0), -1, cv2.LINE_AA)


                # cv2.arrowedLine(frame, (cx1,cy1), (int(pred[0]),int(pred[1])), (255,0,0),1)
            

            if self.showMinimap:
                # Converting coordinates from image to map
                # Just using the larger y value because BBOX center is not were the foot/wheels of the classes are. So center point taken is the center of the bottom line of BBOX
                _, max_y = sorted((y1, y2))
                point_coordinates = self.Minimap_obj.projection_image_to_map((x1+x2)/2, max_y)

                if self.showTrajectory:
                  if frameCount % self.Minimap_obj.updateRate == 0:
                      self.Minimap_obj.realtime_trajectory[tracker_id].append((point_coordinates[0], point_coordinates[1], classID, frameCount))
            
                cv2.circle(minimap_img, point_coordinates, 1, color, -1, cv2.LINE_AA)
                  
                # Plotting the text
                textSize, _ = cv2.getTextSize(str(tracker_id), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                rectangle_start_coord = (point_coordinates[0] + 3, point_coordinates[1] - textSize[1] - 5)
                rectangle_end_coord = (point_coordinates[0] + textSize[0] + 3, point_coordinates[1])
                
                cv2.rectangle(minimap_img, rectangle_start_coord, rectangle_end_coord, color, -1)
                cv2.putText(minimap_img, str(tracker_id), tuple((point_coordinates[0] + 3, point_coordinates[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.textColor, 1, cv2.LINE_AA)
        
        if self.showMinimap:
            frame[self.Minimap_obj.locationMinimap[0][1]:self.Minimap_obj.locationMinimap[1][1], self.Minimap_obj.locationMinimap[0][0]:self.Minimap_obj.locationMinimap[1][0]] = minimap_img

        return frame
