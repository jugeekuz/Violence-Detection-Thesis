import os
import numpy as np
import pandas as pd
import cv2
from collections import deque
import copy
from keras.utils import Sequence
from keras.utils import np_utils


'''
def data_generator(data,batch_size=10,shuffle=True):              
    """
    Yields the next training batch.
    data is an array  [[[frame1_filename,frame2_filename,…frame16_filename],label1], [[frame1_filename,frame2_filename,…frame16_filename],label2],……….].
    """
    num_samples = len(data)
    #if shuffle:
        #data = shuffle_data(data)
    while True:   
        for offset in range(0, num_samples, batch_size):
            #print ('startring index: ', offset) 
            # Get the samples you'll use in this batch
            batch_samples = data[offset:offset+batch_size]
            # Initialise X_train and y_train arrays for this batch
            X_train = []
            y_train = []
            # For each example
            for batch_sample in batch_samples:
                # Load image (X)
                x = batch_sample[0]
                # Read label (y)
                y = batch_sample[1]
                temp_data_list = []
                for img in x:
                    try:
                        img = cv2.imread(img)
                        #apply any kind of preprocessing here
                        #img = cv2.resize(img,(224,224))
                        temp_data_list.append(img)
                    except Exception as e:
                        print (e)
                        print ('error reading file: ',img)                      
                # Add example to arrays
                X_train.append(temp_data_list)
                y_train.append(y)
    
            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            #X_train = np.rollaxis(X_train,1,4)
            y_train = np.array(y_train)
            # create one hot encoding for training in keras
            y_train = np_utils.to_categorical(y_train, 3)
    
            # yield the next training batch            
            yield X_train, y_train
'''


class FrameGenerator(Sequence):
    def __init__(self,path,label='train',batch_size=1,temporal_stride=1,temporal_length=5,target_size=(224,224),n_classes=2,rescale = 1/255.,is_autoencoder=False,shuffle=True):
        #Initializing the values
        self.target_size = target_size
        self.path = path
        self.temporal_stride = temporal_stride
        self.temporal_length = temporal_length
        self.data = pd.DataFrame(self._load_samples(path=self.path,data_cat=label))
        self.batch_size = batch_size
        self.rescale = rescale
        self.list_IDs = np.arange(len(self.data))
        self.n_classes = n_classes
        self.is_autoencoder = is_autoencoder
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self): 
        self.indexes = self.list_IDs  #Load the indexes of the data
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self): 
        return int(np.floor(len(self.data)/self.batch_size))

    def __getitem__(self, index):
        #Generate batch at position 'index' 
        index = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #Generate a temporary list of indexes that forms a batch based on  ##the index selected above.
        list_IDs_temp = [self.list_IDs[k] for k in index]
        #Generate batch
        X,y = self.__data_generation(list_IDs_temp)
        return X,y

    def __transform(self,image):
        ext_img = cv2.resize(image,self.target_size)
        ext_img = ext_img * self.rescale
        return ext_img

    def __data_generation(self,list_IDs_temp):
        X_data = []
        y_data = []
        for _,i in enumerate(list_IDs_temp): #Iterating through each ######################################sequence of frames
            seq_frames = self.data.iloc[i,0]
            y = self.data.iloc[i,1]
            temp_data_list = []
            for img in seq_frames:
                try:
                    image = cv2.imread(img)
                    ext_img = self.__transform(image)
                except Exception as e: 
                    '''Code you'd want to run in case of an exception/err'''
                temp_data_list.append(ext_img)
            X_data.append(temp_data_list) 
            y_data.append(y)
        X = np.array(X_data)  #Converting list to array
        y = np.array(y_data)
        if self.is_autoencoder == True:
            return X, X
        else:
            return X, np_utils.to_categorical(y,num_classes=self.n_classes)

    def _file_generator(self,data_path,data_files,temporal_stride=4,temporal_length=10):
        #for every csv(directory) in list of csv files(directories)
        for f in data_files:
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


    def _load_samples(self,path='data_files',data_cat='train'):
        data_path = os.path.join(path,data_cat)
        data_files = os.listdir(data_path)
        file_gen = self._file_generator(data_path,data_files,self.temporal_stride,self.temporal_length)
        iterator = True
        data_list = []
        while iterator:
            try:
                x,y = next(file_gen)
                x = list(x)
                data_list.append([x,y])
            except Exception as e:
                #print('the exception: ',e)
                iterator = False
                print('Found {} videos belonging to 2 classes.'.format(len(data_list)))
        return data_list