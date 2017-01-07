#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Load the VGG16 network, adopted from https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg16.py
 
VGG-16, a 16-layer model from the paper:
"Very Deep Convolutional Networks for Large-Scale Image Recognition"
Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

Download pretrained weights from:
https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

"""

import cv2
import os
import pickle
import lasagne

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer#, Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer, Pool2DDNNLayer as PoolLayer
from lasagne.nonlinearities import softmax

class VGG16:
    
    def __init__(self, input_var=None, dropout_rate=0.5):
        net = {}
        net['input'] = InputLayer((None, 3, 224, 224), input_var=input_var)
        net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
        net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
        net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
        net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
        net['pool3'] = PoolLayer(net['conv3_3'], 2)
        net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
        net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
        net['pool4'] = PoolLayer(net['conv4_3'], 2)
        net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
        net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
        net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
        net['pool5'] = PoolLayer(net['conv5_3'], 2)
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        net['fc6_dropout'] = DropoutLayer(net['fc6'], p=dropout_rate)
        net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
        net['fc7_dropout'] = DropoutLayer(net['fc7'], p=dropout_rate)
        net['fc8'] = DenseLayer(net['fc7_dropout'], num_units=1000, nonlinearity=None)
        net['prob'] = NonlinearityLayer(net['fc8'], softmax)
        self.net = net
        
    
    def load(self, model_file):
        if not os.path.isfile(model_file):
            raise ValueError('.pkl model not found, it should be available at https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl')
        
        print('openning the .pkl file')
        with open(model_file, 'r') as fid:
            values = pickle.load(fid)
            print('model %s loaded' % values['model name'])
        
        lasagne.layers.set_all_param_values(self.net['prob'], values['param values'])
    
        self.MEAN_BGR = values['mean value'].reshape((1,3,1,1))
        self.classes = values['synset words']
        return self.net        
        
    # Check the model
    def image_classify(self, im_path):
        
        import matplotlib.pyplot as plt
        import numpy as np

        im_size = 256
        crop_size = 224
        im = cv2.imread(im_path)
        k = float(im_size)/np.min((im.shape[1],im.shape[0]))
        im = cv2.resize(im,(int(im.shape[1]*k),int(im.shape[0]*k)))
        center = [int(np.round((im.shape[0]-crop_size)/2.)),int(np.round((im.shape[1]-crop_size)/2.))]
        im = np.asarray(im, dtype='float32').transpose((2,0,1))[None,:,center[0]:center[0]+crop_size,center[1]:center[1]+crop_size]
        plt.imshow(np.squeeze(im[:,::-1,:,:]).transpose((1,2,0))/255)
        plt.show()
        im -= self.MEAN_BGR
        prob = np.array(lasagne.layers.get_output(self.net['prob'], im, deterministic=True).eval())
        plt.plot(prob.T)
        plt.show()
        print(np.argmax(prob)) # 285
        print(self.classes[np.argmax(prob)])
