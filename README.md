# Violence Detection using Deep Learning
## NTUA Thesis  
## Anastasios Diamantis

The following is a thesis about Violence Detection from CCTV footage using Deep Learning.

We test two different approaches in this thesis. One using only frame based approach using transfer learning from InceptionNetV3 architecture
and another using a Two Stream Inflated Conv3D network using frame differences.

-"utilities.py" contains auxiliary functions, mainly aiding the preprocessing of videos for use in our networks.

-"violence_detection_CNN_002.ipynb" is the transfer learning approach

-"violence_detection_Two_Stream_004.ipynb" is the Two Stream Inflated Conv3D network using frame differences.

-"gen/" contains a python library inspired by ImageDataGenerator that serves the purpose of a sequence-of-images generator for use in our networks, including Data Augmentation techniques. 
