# videos
Scripts for video recognition

## Results

### UCF-101

Test acc = video classification accuracy on the UCF-101 [1] test set (split 1), %

#### Models fine tuned on single video frames

Model, fine tuning layers | Test acc      | Parameters
-------                   |:--------:     |--------
VGG-16, last layer        | 73.6          | default
VGG-16, fc layers         | 76.5          | default
VGG-16, fc layers         | 77.0          | dropout=0.8
VGG-16, fc layers         | 77.3          | dropout=0.8, weight decay=5e-4
VGG-16, all layers        | 75.3          | batch size=32, max_iter=40k, step_size=20k
ResNet-50, last layer     | 76.5          | weight decay=5e-4
ResNet-50, all layers     | 79.5          | batch size=32, max_iter=step_size=20k, weight decay=5e-4
ResNet-50, last layer     | 78.5          | pool5 layer modified, weight decay=5e-4
ResNet-50, last layer     | 79.4          | pool5 layer modified, weight decay=5e-4, dropout=0.5
ResNet-50, all layers     | **80.4** (81.4)*  | batch size=32, max_iter=step_size=20k, pool5 layer modified, weight decay=5e-4, dropout=0.5
*all video frames used for prediction (be default only 25 frames with stride 3 are used for prediction)

#### Models fine tuned on sequences of video frames

Model, fine tuning layers           | Test acc      | Parameters
-------                             |:--------:     |--------
ResNet-50, LSTM+last layer ([code](https://github.com/bknyaz/videos/blob/master/ucf101/ucf101_ft_resnet_lstm.ipynb))  | 71.7 (73.1) | 512 units in LSTM, 25 frames for training, 25 (75) frames for prediction
ResNet-50, bnorm+LSTM+last layer ([code](https://github.com/bknyaz/videos/blob/master/ucf101/ucf101_ft_resnet_bnorm_lstm.ipynb))  | 74.0 | 512 units in LSTM, 25 frames for training, 75 frames for prediction

#### Other works

Model, fine tuning layers | Test acc*
-------                   |:--------:
CNN-M-2048, last layer    | 72.7 [2]
Improved DT+FV            | 85.9 [3]
State of the art          | **95.6** [4]
*accuracy averaged over 3 splits in some works

## References

[1] UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild, 2012

[2] Two-Stream Convolutional Networks for Action Recognition in Videos, 2014

[3] Action recognition with improved trajectories, 2013

[4] Deep Temporal Linear Encoding Networks, 2016

