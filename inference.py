import torch
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import math
from torch._C import device
from tqdm import tqdm
from collections import namedtuple, defaultdict
import torch.backends.cudnn as cudnn
import pandas as pd
from datetime import datetime, timedelta
from copy import deepcopy
import time
import shutil

import sys
sys.path.append('./yolo_v5_main_files')
from models.common import DetectMultiBackend, AutoShape
from utils.datasets import LoadImages
from utils.torch_utils import time_sync
from utils.general import LOGGER, non_max_suppression, scale_coords, check_img_size, print_args
from hubconf import custom

from sort_yoloV5 import Sort
from visualizer import Visualizer

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

class Inference():
    def __init__(self, input, model_weights, output, imgSize):        
        # Inference Params
        self.img_size = imgSize
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.agnostic_nms = False
        self.max_det = 300
        self.classes = None # Filter classes

        self.device = torch.device('cuda:0')
        self.half = True
        cudnn.benchmark = True
        
        # Checking input
        if os.path.isfile(input):
            # Further functionality needs to be added for Folder Inference :))
            if input[-4:] in ['.png', '.jpg']:
               self.input = input
               self.inference_mode = 'SingleImage'
            elif input[-4:] in ['.mp4', '.mkv', '.avi']:
                self.input = input
                self.inference_mode = 'Video'
                self.fps = 30
            else:
                print("Invalid input file. The file should be an image or a video !!")
                exit(-1)
        else:
            print("Input file doesn't exist. Check the input path")
            exit(-1)
                 
        # Checking weights file
        if os.path.isfile(model_weights):
            if model_weights[-3:] == '.pt':
                self.model_weights = model_weights
                self.inference_backend = 'PyTorch'
            elif model_weights[-7:] == '.engine':
                self.model_weights = model_weights
                self.inference_backend = 'TensorRT'
            else:
                print(f"Invalid Weights file. {model_weights} does not end with '.engine' or '.pt'")
                exit(-1)
        else:
            print("Model weights file does not exist. Check the weights path")
            exit(-1)
        
        # Checking output
        if output == None:
            self.output = self.input.split('/')[-1]
        else:
            self.output = output
            __output_path_processing = Path(self.output)
            self.file_stem_name = __output_path_processing.stem
            self.parent_directory = __output_path_processing.parents[0]
            self.output_dir_path = self.parent_directory / self.file_stem_name
            if not os.path.exists(self.output_dir_path):
                os.makedirs(self.output_dir_path)
            else:
                shutil.rmtree(self.output_dir_path)           # Removes all the subdirectories!
                os.makedirs(self.output_dir_path)

        # Loading Model
        model = DetectMultiBackend(self.model_weights, device=self.device, dnn=None)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
                      'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                      'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
                      'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
                      'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
                      'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
                      'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
                      'bike', 'hydrant', 'motor', 'rider', 'light', 'sign', 'motor vehicle', 'human face', 'hair drier', 'license plate']
        
        if self.pt:
            model = model.model.half()
        self.model = model

        # Initialize Tracker
        self.Objtracker = Sort(max_age=40, min_hits=7, iou_threshold=0.3)
        self.Objtracker.reset_count()
        
        # Parameters for velocity estimation
        self.trackDict = defaultdict(list)

        self.runInference()
    
    def UpdateTracker(self, pred):
        if len(pred) > 0:
            dets = []
            for items in pred:
                dets.append(items[:].tolist())
        
            dets = np.array(dets)
            self.tracker = self.Objtracker.update(dets)
        else:
            self.tracker = self.Objtracker.update()
        
    def Trajectory_points(self, trajectory_array):    
        for detection in self.tracker:
            class_id = detection[5]
            center_x = (detection[0] + detection[2])/2 

            if class_id in (0,1,3,16):
                _, max_y = sorted((detection[1], detection[3]))
            elif class_id in (2,5):
                max_y = (detection[1] + detection[3])/2     # Center of bbox for classes other than Escooter, Cyclist, and Pedestrian

            trackID = int(detection[9])
            self.trackDict[trackID].append((int(center_x), int(max_y)))
            
            if len(self.trackDict[trackID]) > 10: 
                output_array = np.append(detection, [self.trackDict[trackID][-2][0], self.trackDict[trackID][-2][1], self.trackDict])
                trajectory_array.append(output_array)
                del self.trackDict[trackID][0]

        return trajectory_array


    def runInference(self):
        dataset = LoadImages(self.input, img_size=self.img_size, stride=self.stride, auto=self.pt and not self.jit)
        bs = 1
        vid_path, vid_writer = None, None

        Visualize = Visualizer()
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        framecount = 0
        time_start = time_sync()
        for path, im, im0, vid_cap, s, videoTimer in dataset:
            framecount += 1
            if framecount < -1:
                continue
            elif framecount > 72000:
                break
            # storing_output = {}
            # storing_output["Video_Internal_Timer"]= videoTimer

            # Image Preprocessing for inference
            t1 = time_sync()
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = self.model(im, augment=False, visualize=False)
            t3 = time_sync()
            dt[1] += t3 - t2
            if self.pt:
                pred = pred[0]

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)[0]
            t4 = time_sync()
            dt[2] += t4 - t3

            # Process predictions
            seen += 1

            s += '%gx%g ' % im.shape[2:] 

            if len(pred):
                # Rescale boxes from img_size to im0 size
                pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], im0.shape).round()

                # Print results
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string    
        
            # Save the images or videos
            if self.inference_mode == 'SingleImage':
                self.frame = Visualize.drawBBOX(pred, im0)
                cv2.imwrite(self.output, self.frame)
            
            elif self.inference_mode == 'Video':
                # Update the tracker
                self.UpdateTracker(pred)
                
                trajectory_array = []
                stored_trajectory = self.Trajectory_points(trajectory_array)

                if len(self.tracker) > 0:
                    frame = Visualize.drawTracker(stored_trajectory, im0, framecount)
                elif len(pred) > 0:
                    frame = Visualize.drawBBOX(pred, im0, framecount)
                else:
                    frame = Visualize.drawEmpty(im0, framecount)
                   
                t5 = time_sync()
                dt[3] += t5 - t4
                print(f'{s}Done. ({1/(t3 - t2):.3f}fps)(Post: {((t5 - t4)*1000):.3f}ms)')

                if self.save_infer_video:
                    if framecount == 1:  # new video
                        final_path = os.path.join(self.output_dir_path, self.output.split('\\')[-1])
                        print(final_path)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(final_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
                    vid_writer.write(frame)      

        if self.inference_mode == 'Video':    
            vid_writer.release()

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.3fms NMS per image at shape {(1, 3, *im.shape[2:])}, %.1fms Post-processing' % t)
        time_end = time_sync()
        print(f'Total time for inference (including pre and post-processing): {round(time_end-time_start, 2)}s')
        print(f'Average total fps: {round(framecount/round(time_end-time_start, 2), 2)}fps')
        


    def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', type=str, default=None, help=['path to input file(s)', '.MP4/.mkv/.png/.jpg/.jpeg'])
        parser.add_argument('--model_weights', type=str, default=None, help='model\'s weights path(s)')
        parser.add_argument('--output', type=str, default=None, help=['path to save result(s)', '.MP4/.mkv/.png/.jpg/.jpeg'])
        parser.add_argument('--imgSize','--img','--img_size', nargs='+', type=int, default=[480], help='inference size h,w')
               
        opt = parser.parse_args()
        opt.imgSize *= 2 if len(opt.imgSize) == 1 else 1
        print_args(FILE.stem, opt)
        return opt

    def main(opt):
        Inference(**vars(opt))

    
if __name__ == "__main__":
    opt = Inference.parse_opt()
    print("---- FLIR Inference ----")

    Inference.main(opt)
    print("\n")