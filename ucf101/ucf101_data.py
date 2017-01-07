#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
UCF-101 data loader

@author: Boris Knyazev
"""
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2

class DataLoader(object):

    def __init__(self, data_dir, split=1, shuffle_train=False, rand_seed=11):
        
        self.data_dir = data_dir
        self.split = split
        self.shuffle_train = shuffle_train
        self.random  = np.random.RandomState(rand_seed) # to reproduce
        
        labels_dir = data_dir + '/ucfTrainTestlist/'
        with open(labels_dir + '/trainlist0%d.txt' % split, 'r') as f:
            samples = f.readlines()
        
        train_labels = []
        train_videos = []
        for s in samples:
            path, label = s.split()
            train_labels.append(int(label)-1)
            full_path = data_dir + '/UCF-101/' + path
            if not os.path.isfile(full_path):
                raise ValueError('corrupted dataset or invalid folder structure')
            train_videos.append(full_path)
        
        self.train_labels = np.asarray(train_labels,dtype='uint8')
        self.train_videos = train_videos
        if shuffle_train:
            ids = self.random.permutation(len(self.train_labels))
            self.train_labels = self.train_labels[ids]
            self.train_videos = [self.train_videos[i] for i in ids]
        
        
        with open(labels_dir + 'testlist0%d.txt' % split, 'r') as f:
            samples = f.readlines()
        
        with open(labels_dir + 'classInd.txt', 'r') as f:
            classes_str = f.readlines()    
        
        classes = []
        for s in classes_str:
            label, name = s.split()
            classes.append(name)
       
        self.classes = classes
       
        test_labels = []
        test_videos = []
        for s in samples:
            path = s.split()[0]
            full_path = data_dir + '/UCF-101/' + path
            if not os.path.isfile(full_path):
                raise ValueError('corrupted dataset or invalid folder structure')
            test_videos.append(full_path)
            test_labels.append(np.where([c == path.split('/')[0] for c in classes])[0][0])
         
        self.test_videos = test_videos
        self.test_labels = np.asarray(test_labels,dtype='uint8')
        
        # Check dataset consistency
        assert(len(self.train_labels) == len(train_videos))
        assert(len(self.test_labels) == len(test_videos))
        assert(len(np.unique(self.train_labels)) == len(np.unique(self.train_labels)) == len(self.classes) == 101)
        
        sets = ['Train','Test']
        for labels,set_name in zip([train_labels,test_labels],sets):
            hist, bin_edges = np.histogram(labels, bins=len(classes))
            for i,h in enumerate(hist):
                print('%s : %d samples' % (classes[i],h))
                
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.bar(bin_edges[:-1], hist)
            ax.set_xlabel('Action')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of %s action labels' % set_name)
            plt.show()
            
        print('SPLIT=%d: %d training samples, %d test samples, %d classes' % (self.split,len(self.train_labels),len(self.test_labels),len(self.classes)))
    
    
    # Gets a batch of samples from the dataset
    def get_batch_videos(self, iterable_samples, iterable_labels, batch_size):
        l = len(iterable_samples)
        for ndx in range(0, l, batch_size):
            yield iterable_samples[ndx:min(ndx + batch_size, l)], iterable_labels[ndx:min(ndx + batch_size, l)]
    
    
    def im_crop(self, im, MEAN_BGR, train=False):
        # resize
        k = 256./np.min((im.shape[1],im.shape[0]))
        im = cv2.resize(im,(int(im.shape[1]*k),int(im.shape[0]*k))).astype(np.float32)
        
        # crop
        crop_size = 224
        if train:
            center = [np.random.randint(0, high=im.shape[0]-crop_size),np.random.randint(0, high=im.shape[1]-crop_size)]
            if np.random.rand() > 0.5: # random flip
                im = im[:,::-1,:] # height x width x channels
        else:
            center = [int(np.round((im.shape[0]-crop_size)/2.)),int(np.round((im.shape[1]-crop_size)/2.))]
        im = im.transpose((2,0,1))[None,:,center[0]:center[0]+crop_size,center[1]:center[1]+crop_size]
        return im-MEAN_BGR
    
    # Gets random frame (random crop) from each video
    def get_frames_train(self, frames_dir, videos, MEAN_BGR):
        frames = np.zeros((len(videos),3,224,224),dtype='float32')
        for v_id,video in enumerate(videos):
            video_names = video.split('/')
            frames_dir_video = '%s/%s/%s/' % (frames_dir,video_names[-2],video_names[-1][:-4])
            n_frames = len(os.listdir(frames_dir_video))
            frame_path = '%s/%d.jpg' % (frames_dir_video,np.random.permutation(n_frames)[0]) # get random frame
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError('frame invalid')
            frames[v_id] = self.im_crop(frame, MEAN_BGR, train=True)
            
        return frames

    # Gets at most N_FRAMES frames (central crops) from each video
    def get_frames_val(self, frames_dir, videos, MEAN_BGR, video_labels, STRIDE=10, N_FRAMES=5):
        frames = []
        labels = []
        for v_id,video in enumerate(videos):
            video_names = video.split('/')
            frames_dir_video = '%s/%s/%s/' % (frames_dir,video_names[-2],video_names[-1][:-4])
            n_frames = len(os.listdir(frames_dir_video))
            frames.append([])
            labels.append([])
            for frame_id in range(0,n_frames,STRIDE)[:N_FRAMES]:
                frame_path = '%s/%d.jpg' % (frames_dir_video,frame_id)
                frame = cv2.imread(frame_path)
                if frame is None:
                    raise ValueError('frame invalid')
                frames[-1].append(self.im_crop(frame, MEAN_BGR).astype(np.float32))
                labels[-1].append(video_labels[v_id])

            frames[-1] = np.concatenate(frames[-1])
            labels[-1] = np.asarray(labels[-1])
            
        return frames, labels

if __name__ == "__main__":         
    dloader = DataLoader(shuffle_train=True)