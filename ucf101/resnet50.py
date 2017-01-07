#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Load the ResNet-50 network, adopted from https://github.com/Lasagne/Recipes/blob/master/modelzoo/resnet50.py

Pool2DLayer replaced with Pool2DDNNLayer, ignore_border is set to True to use cuDNN and GPU
(see warning at https://github.com/Theano/Theano/blob/master/theano/tensor/signal/pool.py)

ResNet-50, network from the paper:
"Deep Residual Learning for Image Recognition"
http://arxiv.org/pdf/1512.03385v1.pdf
License: see https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE

Download pretrained weights from:
https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/resnet50.pkl
 
"""

import os, cv2, pickle

import lasagne
from lasagne.layers import InputLayer, BatchNormLayer, NonlinearityLayer, ElemwiseSumLayer, DenseLayer
#from lasagne.layers import Conv2DLayer as ConvLayer
#from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import rectify, softmax

from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer, Pool2DDNNLayer as PoolLayer


class Resnet50:
    
    def __init__(self, input_var=None):
        self.build_model(input_var)

    def build_simple_block(self, incoming_layer, names,
                           num_filters, filter_size, stride, pad,
                           use_bias=False, nonlin=rectify):
        """Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)
    
        Parameters:
        ----------
        incoming_layer : instance of Lasagne layer
            Parent layer
    
        names : list of string
            Names of the layers in block
    
        num_filters : int
            Number of filters in convolution layer
    
        filter_size : int
            Size of filters in convolution layer
    
        stride : int
            Stride of convolution layer
    
        pad : int
            Padding of convolution layer
    
        use_bias : bool
            Whether to use bias in conlovution layer
    
        nonlin : function
            Nonlinearity type of Nonlinearity layer
    
        Returns
        -------
        tuple: (net, last_layer_name)
            net : dict
                Dictionary with stacked layers
            last_layer_name : string
                Last layer name
        """
        net = []
        net.append((
                names[0],
                ConvLayer(incoming_layer, num_filters, filter_size, pad, stride,
                          flip_filters=False, nonlinearity=None) if use_bias
                else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                               flip_filters=False, nonlinearity=None)
            ))
    
        net.append((
                names[1],
                BatchNormLayer(net[-1][1])
            ))
        if nonlin is not None:
            net.append((
                names[2],
                NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
            ))
    
        return dict(net), net[-1][0]
    
    
    def build_residual_block(self, incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                             upscale_factor=4, ix=''):
        """Creates two-branch residual block
    
        Parameters:
        ----------
        incoming_layer : instance of Lasagne layer
            Parent layer
    
        ratio_n_filter : float
            Scale factor of filter bank at the input of residual block
    
        ratio_size : float
            Scale factor of filter size
    
        has_left_branch : bool
            if True, then left branch contains simple block
    
        upscale_factor : float
            Scale factor of filter bank at the output of residual block
    
        ix : int
            Id of residual block
    
        Returns
        -------
        tuple: (net, last_layer_name)
            net : dict
                Dictionary with stacked layers
            last_layer_name : string
                Last layer name
        """
        simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']
    
        net = {}
    
        # right branch
        net_tmp, last_layer_name = self.build_simple_block(
            incoming_layer, map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern),
            int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter), 1, int(1.0/ratio_size), 0)
        net.update(net_tmp)
    
        net_tmp, last_layer_name = self.build_simple_block(
            net[last_layer_name], map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern),
            lasagne.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
        net.update(net_tmp)
    
        net_tmp, last_layer_name = self.build_simple_block(
            net[last_layer_name], map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern),
            lasagne.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor, 1, 1, 0,
            nonlin=None)
        net.update(net_tmp)
    
        right_tail = net[last_layer_name]
        left_tail = incoming_layer
    
        # left branch
        if has_left_branch:
            net_tmp, last_layer_name = self.build_simple_block(
                incoming_layer, map(lambda s: s % (ix, 1, ''), simple_block_name_pattern),
                int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
                nonlin=None)
            net.update(net_tmp)
            left_tail = net[last_layer_name]
    
        net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
        net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify)
    
        return net, 'res%s_relu' % ix
    
    
    def build_model(self, input_var=None):
        net = {}
        net['input'] = InputLayer((None, 3, 224, 224),input_var=input_var)
        sub_net, parent_layer_name = self.build_simple_block(
            net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
            64, 7, 3, 2, use_bias=True)
        net.update(sub_net)
        net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=True) 
        # with ignore_border=False 1sec and correct output
        block_size = list('abc')
        parent_layer_name = 'pool1'
        for c in block_size:
            if c == 'a':
                sub_net, parent_layer_name = self.build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
            else:
                sub_net, parent_layer_name = self.build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c)
            net.update(sub_net)
    
        block_size = list('abcd')
        for c in block_size:
            if c == 'a':
                sub_net, parent_layer_name = self.build_residual_block(
                    net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c)
            else:
                sub_net, parent_layer_name = self.build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c)
            net.update(sub_net)
    
        block_size = list('abcdef')
        for c in block_size:
            if c == 'a':
                sub_net, parent_layer_name = self.build_residual_block(
                    net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c)
            else:
                sub_net, parent_layer_name = self.build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c)
            net.update(sub_net)
    
        block_size = list('abc')
        for c in block_size:
            if c == 'a':
                sub_net, parent_layer_name = self.build_residual_block(
                    net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
            else:
                sub_net, parent_layer_name = self.build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
            net.update(sub_net)
        
        net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                                 mode='average_exc_pad', ignore_border=True) # with ignore_border=False 0.750, ignore_border=None -> warning
        net['fc1000'] = DenseLayer(net['pool5'], num_units=1000, nonlinearity=None)
        net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax)
        
        self.net = net
        
        return net
        
    def load(self, model_file):
        if not os.path.isfile(model_file):
            raise ValueError('.pkl model not found, it should be available at https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/resnet50.pkl')
        
        print('openning the .pkl file')
        with open(model_file, 'r') as fid:
            values = pickle.load(fid)
            print('model %s loaded' % 'Resnet50')
        
        lasagne.layers.set_all_param_values(self.net['prob'], values['values'])
    
        self.MEAN_BGR = values['mean_image'].reshape((1,3,224,224))
        self.classes = values['synset_words']
        return self.net        
        
        
    # Check the model
    def image_classify(self, im_path=None):

        import matplotlib.pyplot as plt
        import numpy as np, io
        if im_path is None:
            import urllib
            url = "http://farm1.static.flickr.com/8/11912062_a1dda4fa83.jpg"
            ext = url.split('.')[-1]
            im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)
        else:         
            ext = im_path.split('.')[-1]
            im = plt.imread(im_path, ext)
            
        im_size = 256
        crop_size = 224
        
        k = float(im_size)/np.min((im.shape[1],im.shape[0]))
        im = cv2.resize(im[:,:,::-1],(int(im.shape[1]*k),int(im.shape[0]*k)))
        center = [int(np.round((im.shape[0]-crop_size)/2.)),int(np.round((im.shape[1]-crop_size)/2.))]
        im = np.asarray(im, dtype='float32').transpose((2,0,1))[None,:,center[0]:center[0]+crop_size,center[1]:center[1]+crop_size]
        plt.imshow(np.squeeze(im[:,::-1,:,:]).transpose((1,2,0))/255)
        plt.show()
        im -= self.MEAN_BGR
        prob = np.squeeze(lasagne.layers.get_output(self.net['prob'], im, deterministic=True).eval())
        plt.plot(prob)
        plt.show()
        ind = np.argsort(prob)
        print(ind[-1]) # 235
        print(self.classes[ind[-1]],self.classes[ind[-2]])
        print(prob[ind[-1]],prob[ind[-2]])
#        assert(abs(prob[ind[-1]]-0.77) < 0.01)
        
#        prob = np.array(lasagne.layers.get_output(self.net['res5c_relu'], im, deterministic=True).eval())
#        print(prob.shape) # (1, 2048, 7, 7)
#        prob = np.array(lasagne.layers.get_output(self.net['pool5'], im, deterministic=True).eval())
#        print(prob.shape) # (1, 2048, 1, 1)
