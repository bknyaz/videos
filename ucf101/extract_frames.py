# -*- coding: utf-8 -*-
"""
Extracts and saves all frames from UCF101 videos using OpenCV
It is a necessary step before running ucf101_ft_vgg16.py
My resulted folder (UCF101_folder/raw) contains 2483295 .jpg files (i.e., 186 frames per video on average) 
which in total take about 56GB of disk space

Example:
python extract_frames.py UCF101_folder

where UCF101_folder - folder in which folders ucfTrainTestlist and UCF-101 
(with a folder for each action class inside) must be available

@author: Boris Knyazev
"""

import numpy as np
import time
import os, sys
import cv2

from ucf101_load import DataLoader

dloader = DataLoader(data_dir=sys.argv[1], shuffle_train=False)
    
frames_dir = dloader.data_dir + '/raw/'

if not os.path.isdir(frames_dir):
    os.mkdir(frames_dir)

start = time.time()
    
total_frames = 0
video_id = 0
all_videos = np.concatenate((dloader.train_videos,dloader.test_videos))
for video in all_videos:

    video_names = video.split('/')
    frames_dir_video_parent = '%s/%s/' % (frames_dir,video_names[-2])
    frames_dir_video = '%s/%s/' % (frames_dir_video_parent,video_names[-1][:-4])
  
    if not os.path.isdir(frames_dir_video_parent):
        os.mkdir(frames_dir_video_parent)
        
    if not os.path.isdir(frames_dir_video):
        os.mkdir(frames_dir_video)
      
    cap = cv2.VideoCapture(video)
    frame_id = 0
    while True:
        read,frame = cap.read()
        if not read:
            break
        
        frame_path = frames_dir_video + '%d.jpg' % frame_id
          
        if os.path.isfile(frame_path):
            s = '(skipped) '
        else:
            s = ''
            if not cv2.imwrite(frame_path, frame):
                raise ValueError('frame not saved')
        
        end = time.time()
        
        frame_id += 1
        total_frames += 1
        
        print('%savg fps=%.3f, frame=%d (total=%d), video(%d/%d)=%s, target_dir=%s' % (s.upper(),total_frames/(end-start),frame_id,total_frames+1,video_id,len(all_videos),video_names[-1],frames_dir_video))
        
    cap.release()
    video_id += 1
