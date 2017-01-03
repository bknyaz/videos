# -*- coding: utf-8 -*-
"""

Fine tuning of the VGG16 network on the UCF101 video (action) classification dataset
"Khurram Soomro, Amir Roshan Zamir and Mubarak Shah, 
UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild., 
CRCV-TR-12-01, November, 2012."

Acc: Video classification accuracy on the UCF-101 test set (split 1)

Last layer fine tuning: Acc 73.6%
FC layers fine tuning: Acc 76.5%
All layers fine tuning: Acc 

@author: Boris Knyazev
"""

import numpy as np
import time
import os
import argparse

import pickle
import theano, lasagne
import theano.tensor as T

from lasagne.layers import NonlinearityLayer, DenseLayer
from lasagne.nonlinearities import softmax
from timeit import default_timer as timer

from ucf101_data import DataLoader
from vgg16 import VGG16

start = timer()

#####################
### Parse options ###
#####################

parser = argparse.ArgumentParser(description='Fine tuning of the VGG16 network on the video frames data')
parser.add_argument('-M','--model', default='/home/boris/Project/models/vgg/vgg16.pkl', help='location of the .pkl network')
parser.add_argument('-L','--layers', default='all', help='layers to be fine tuned (last, fc, all)')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate: lr = 0.01-0.0001 can be a reasonable choice')
parser.add_argument('--step_size', default=20000, type=int, help='number of steps after which lr will be decreased by lrdec')
parser.add_argument('--max_iter', default=40000, type=int, help='number of learning iterations')
parser.add_argument('--val_step', default=400, type=int, help='number of steps after which validate the network')
parser.add_argument('--lrdec', default=0.1, type=float, help='learning rate decay, multiplies lr each step_size iterations')
parser.add_argument('--wdec', default=1e-5, type=float, help='weight decay')
parser.add_argument('--drop', default=0.5, type=float, help='dropout rate')
parser.add_argument('-B','--bs', default=32, type=int, help='batch size')
parser.add_argument('-s','--stride', default=3, help='stride between frames during validation and testing')
parser.add_argument('-n','--n_frames', default=25, help='number of frames per video during validation and testing')
parser.add_argument('--plot', default=True, type=bool, help='save plots, figures, etc.')
parser.add_argument('--save', default=10000, type=int, help='step after which save snapshots, 0 - do not save')
parser.add_argument('-D','--dir', default='/home/boris/Project/data/videos/UCF101', help='folder in which folders ucfTrainTestlist and UCF-101 must be available')
parser.add_argument('--results_dir', default='results', help='save plots, figures, models, etc. to this directory')

print('experiment parameters:')
for a in vars(parser.parse_args()).items():
    print('%s = %s' % (a[0],a[1]))

args = parser.parse_args()    
    
if args.results_dir == '':
    raise ValueError('specify correct results directory')
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

dloader = DataLoader(args.dir, shuffle_train=True)
frames_dir = dloader.data_dir + '/raw/'

##########################
### Define the network ###
##########################

input_var = T.tensor4('inputs')
model = VGG16(input_var, dropout_rate=args.drop)
model.load(args.model)
#model.image_classify('/home/boris/Project/3rd_party/caffe/examples/images/cat.jpg')
net = model.net
last_layer = 'prob'
net['fc8'] = DenseLayer(net['fc7_dropout'], num_units=len(dloader.classes), nonlinearity=None)
net[last_layer] = NonlinearityLayer(net['fc8'], softmax)
if args.layers != 'all':
    for layer in net.items():
        if (args.layers == 'last' and layer[0] == 'fc8') or (args.layers == 'fc' and layer[0].find('fc') >= 0):
            continue
        try:
            layer[1].params[layer[1].W].remove("trainable")
            layer[1].params[layer[1].b].remove("trainable")
        except:
            continue

print('building the network...')
target_var = T.ivector('targets')
prediction_val = T.clip(lasagne.layers.get_output(net[last_layer], deterministic=True), 1e-10, 1.0 - 1e-10)
loss_val = lasagne.objectives.categorical_crossentropy(prediction_val, target_var)
val_fn = theano.function([input_var, target_var], [loss_val, prediction_val])

prediction_train = T.clip(lasagne.layers.get_output(net[last_layer]), 1e-10, 1.0 - 1e-10)
loss_train_all = lasagne.objectives.categorical_crossentropy(prediction_train, target_var)
loss_train = loss_train_all.mean()
weightsl2 = lasagne.regularization.regularize_network_params(net[last_layer], lasagne.regularization.l2)
loss_train += args.wdec * weightsl2

lr_var = theano.shared(np.array(args.lr, dtype=theano.config.floatX))
params = lasagne.layers.get_all_params(net[last_layer], trainable=True)
print('%d trainable pairs of weight matrix W and bias vector b ' % len(params))
updates = lasagne.updates.nesterov_momentum(loss_train, params, learning_rate=lr_var, momentum=0.9)
train_fn = theano.function([input_var, target_var], [loss_train, prediction_train], updates=updates)

n_val = 1000
train_data = (dloader.train_videos[:-n_val], dloader.train_labels[:-n_val])
val_data = (dloader.train_videos[-n_val:], dloader.train_labels[-n_val:])

################
### TRAINING ###
################

loss_train_vs_iter, loss_val_vs_iter, loss_val_vs_iter, err_val_vs_iter, err_val_videos_vs_iter = [],[],[],[],[]
data_load_time, train_iter_time = 0, 0
train_iter = dloader.get_batch_videos(train_data[0], train_data[1], args.bs)
it = 1
while it <= args.max_iter:
    
    if it % args.step_size == 0:
        lr_var.set_value(lr_var.get_value() * np.float32(0.1))
        print('lr = %1.7f' % lr_var.get_value())            

    try:
        batch = train_iter.next()
    except StopIteration:
        ind = np.random.permutation(len(train_data[1]))
        train_data = ([train_data[0][i] for i in ind], train_data[1][ind])
        train_iter = dloader.get_batch_videos(train_data[0], train_data[1], args.bs)
        batch = train_iter.next()
    
    start = time.time()
    frames = dloader.get_frames_train(frames_dir, batch[0], model.MEAN_BGR)
    data_load_time += time.time() - start
    start = time.time()
    loss, prediction = train_fn(frames, batch[1]) # update weights, make predictions for each frame
    train_iter_time += time.time() - start
    loss_train_vs_iter.append((it,loss))
    if it % min((args.val_step,20)) == 0:
        print('it (batch) %d, train loss %.4f, load data/iter %.3f, train time/iter %.3f' % (it, loss, 
              data_load_time/20., train_iter_time/20.))
        data_load_time, train_iter_time = 0, 0
        
       
    ##################
    ### VALIDATION ###
    ##################
   
    if it % args.val_step == 0 or it <= 1:
        start = time.time()
        val_loss, val_predictions, val_predictions_videos, frame_labels = [],[],[],[]
        for batch in dloader.get_batch_videos(val_data[0], val_data[1], args.bs):
            frames, labels = dloader.get_frames_val(frames_dir, batch[0], model.MEAN_BGR, batch[1])
            for f,_ in enumerate(frames):
                loss, prediction = val_fn(frames[f], labels[f]) # predict labels for each frame
                val_loss.append(loss)
                val_predictions.append(prediction)
                val_predictions_videos.append(np.mean(prediction,axis=0)) # get labels for each video
            frame_labels.append(np.concatenate(labels))
        
        frame_labels = np.concatenate(frame_labels)
        loss_val_vs_iter.append((it,np.mean(np.concatenate(val_loss))))
        val_predictions = np.concatenate(val_predictions).reshape(-1,len(dloader.classes))
        val_err_frames = 1 - np.mean(np.equal(np.argmax(val_predictions, axis=1), frame_labels))
        err_val_vs_iter.append((it,val_err_frames))
        val_predictions_videos = np.concatenate(val_predictions_videos).reshape(-1,len(dloader.classes))
        val_err_videos = 1 - np.mean(np.equal(np.argmax(val_predictions_videos, axis=1), val_data[1]))
        err_val_videos_vs_iter.append((it,val_err_videos))
        print('it (batch) %d, val time %.3f, val loss %.4f, val error %.4f, val error (videos) %.4f' % 
        (it, time.time() - start, loss_val_vs_iter[-1][1], val_err_frames, val_err_videos))
        
        if args.plot:
            # plot 4 graphs: training loss, validation loss (frames), validation error (frames), validation error (videos)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('iter')
            ax.set_ylabel('loss')
            ax.set_title('Loss vs iter')
            plt.plot([t[0] for t in loss_train_vs_iter], [t[1] for t in loss_train_vs_iter],'r-+', label="Training loss")    
            plt.plot([t[0] for t in loss_val_vs_iter], [t[1] for t in loss_val_vs_iter],'b-+', label="Validation loss")
            plt.legend(loc='upper right')
            plt.savefig('%s/loss_vs_iter.png' % args.results_dir, dpi=fig.dpi)
            plt.show()
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('iter')
            ax.set_ylabel('error')
            ax.set_title('Error vs iter')
            plt.plot([t[0] for t in err_val_vs_iter], [t[1] for t in err_val_vs_iter],'g--', label="Validation error")
            plt.plot([t[0] for t in err_val_videos_vs_iter], [t[1] for t in err_val_videos_vs_iter],'c-+', label="Validation error (videos)")
            plt.legend(loc='upper right')
            plt.savefig('%s/error_vs_iter.png' % args.results_dir, dpi=fig.dpi)
            plt.show()
        
        print('saving results')
        np.save('%s/losses_errors_labels' % args.results_dir, (loss_train_vs_iter, loss_val_vs_iter, 
                                                               err_val_vs_iter, err_val_videos_vs_iter))

    if it % args.save == 0:
        print('saving model')
        values = lasagne.layers.get_all_param_values(net[last_layer])
        with open('%s/snapshot_%d.pkl' % (args.results_dir,it) , 'w') as f:
            pickle.dump(values, f, protocol=pickle.HIGHEST_PROTOCOL)
        del values
        
    it += 1



################
### TESTING ###
################
print('Testing performance of the trained model')
start = time.time()
test_loss, test_predictions, test_predictions_videos, frame_labels = [],[],[],[]
for batch in dloader.get_batch_videos(dloader.test_videos, dloader.test_labels, args.bs):
    frames, labels = dloader.get_frames_val(frames_dir, batch[0], model.MEAN_BGR, batch[1],STRIDE=args.stride,N_FRAMES=args.n_frames)
    for f,_ in enumerate(frames):
        loss, prediction = val_fn(frames[f], labels[f]) # predict labels for each frame
        test_loss.append(loss)
        test_predictions.append(prediction)
        test_predictions_videos.append(np.mean(prediction,axis=0)) # get labels for each video
    frame_labels.append(np.concatenate(labels))

frame_labels = np.concatenate(frame_labels)
test_loss = np.mean(np.concatenate(test_loss))
test_predictions = np.concatenate(test_predictions).reshape(-1,len(dloader.classes))
test_err_frames = 1 - np.mean(np.equal(np.argmax(test_predictions, axis=1), frame_labels))
test_predictions_videos = np.concatenate(test_predictions_videos).reshape(-1,len(dloader.classes))
test_err_videos = 1 - np.mean(np.equal(np.argmax(test_predictions_videos, axis=1), dloader.test_labels))
print('test time %.3f, test loss %.4f, test error %.4f, test error (videos) %.4f (acc %.2f)' % (time.time() - start, test_loss, 
                                                                    test_err_frames, test_err_videos, (1-test_err_videos)*100))
np.save('%s/loss_errors_labels_test' % args.results_dir, (test_predictions, test_loss, test_err_frames, 
                                                          test_predictions_videos, test_err_videos))