import io
import os
import warnings
from pathlib import Path
import pandas as pd
import copy
from collections import deque
import numpy as np
import cv2


#generates sequences of directories
def _sequence_generator_from_CSV(data_path,temporal_length,temporal_stride):
    #for every csv(directory) in list of csv files(directories)
    csv_files = os.listdir(data_path)

    for f in csv_files:
        tmp_df = pd.read_csv(os.path.join(data_path,f))
        label_list = list(tmp_df["Label"])
        total_images = len(label_list)

        #Just in case we ask for temporal length bigger than all the images we have
        if total_images >= temporal_length:
            num_samples = int((total_images-temporal_length)/temporal_stride) + 1
            #print('num of samples from vide seq--{}:  {}'.format(f,num_samples))
            img_list = list(tmp_df['FileName'])
        else:
            print('num of frames is less than temporal length; discarding this file--{}'.format(f))
            continue

        start_frame = 0
        #we have a rolling window of size==temporal_length and each iteration
        #we add $temporal_strides # of elements and pop after the first 3 
        samples = deque()
        samp_count=0
        for img in img_list:
            samples.append(img)
            if len(samples)==temporal_length:
                samples_c= copy.deepcopy(samples)
                samp_count+=1
                for t in range(temporal_stride):
                    samples.popleft()
                yield samples_c,label_list[0]

#uses CSV generator to create file of all dirs with labels
def _load_samples_from_CSV(path,temporal_length,temporal_stride):
    data_path = path
    file_gen = _sequence_generator_from_CSV(data_path,temporal_length,temporal_stride)
    iterator = True
    sequence_filenames = []
    sequence_classes = []
    while iterator:
        try:
            x,y = next(file_gen)
            x = list(x)
            sequence_filenames.append(x)
            sequence_classes.append(y)
        except Exception as e:
            #print('the exception: ',e)
            iterator = False
            #print('Found {} sequences of {} images each.'.format(len(sequence_filenames), temporal_length))
    return sequence_filenames, sequence_classes


def _load_sequence(filepaths,target_size,prefix=None,**kwargs):
    
    sequence_shape = (len(filepaths),) + target_size
    sequence = np.zeros((sequence_shape))    
    for i, imgpath in enumerate(filepaths):
        if prefix:
            imgpath = os.path.join(prefix,imgpath)
        try:
            img = cv2.imread(imgpath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,target_size[0:2])
        except Exception as e:
            print('Couldnt read image {}, throwing exception{}'.format(imgpath,e))
            img = np.zeros(target_size,np.uint8)

        sequence[i] = img
    return sequence
        


