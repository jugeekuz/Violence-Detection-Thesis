import os
import numpy as np
import pandas as pd
import cv2
from collections import deque
import copy
from tensorflow.keras.utils import Sequence
from keras.utils import np_utils
import random
from tensorflow.keras.preprocessing.image import apply_brightness_shift, apply_affine_transform
# horizontal flip
# random brightness
# rotation range


class FrameGenerator(Sequence):
    def __init__(self,
                path,
                label='train',
                batch_size=1,
                temporal_stride=1,
                temporal_length=5,
                target_size=(224,224),
                n_classes=2,
                #Augmentations
                rescale = 1/255.,
                data_augmentation = False,
                jitter_prob = 0,
                brightness_range = [1.0,1.0],
                rotation_range = 0,
                horizontal_flip = False,

                append_flows=False,
                is_autoencoder=False,
                shuffle=True,
                vertical_flip=False):
        #Initializing the values
        self.target_size = target_size
        self.path = path
        self.temporal_stride = temporal_stride
        self.temporal_length = temporal_length
        self.data = pd.DataFrame(self._load_samples(path=self.path,data_cat=label))
        self.batch_size = batch_size
        self.rescale = rescale

        self.vertical_flip = vertical_flip
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self._data_augmentation = data_augmentation

        self.list_IDs = np.arange(len(self.data))
        self.n_classes = n_classes
        self.is_autoencoder = is_autoencoder
        self.append_flows=append_flows
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
    
    # Data augmentation
    # Per Image
    def __transform(self,image):
        ext_img = cv2.resize(image,self.target_size)
        #we cant rescale here
        #ext_img = ext_img * self.rescale
        return ext_img

    # Per Sequence 
    def __horizontal_flip(self,frame_seq:np.ndarray)->np.ndarray:
        flip = random.sample([True,False],1)
        if not flip:
            return frame_seq

        for i in range(len(frame_seq)):
            frame_seq[i] = cv2.flip(frame_seq[i],1)
        print("Augmentation horizontal flip: Done!")

        return frame_seq
    # We have to rescale before this
    def __random_rotation(frame_seq:np.ndarray, rg, prob=1, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0., interpolation_order=1)->np.ndarray:
        if np.random.rand() > prob:
            return frame_seq
        theta = np.random.uniform(-rg, rg)
        for i in range(np.shape(frame_seq)[0]):
            x = apply_affine_transform(frame_seq[i, :, :, :], theta=theta,row_axis=row_axis,col_axis=col_axis,channel_axis=channel_axis,
                                        fill_mode=fill_mode, cval=cval,
                                        order=interpolation_order)
            frame_seq[i] = x
        print("Augmentation rotation: Done!")
        
        return frame_seq

    #we have to rescale after this not before
    def __color_jitter(self,frames_seq:np.ndarray,prob=1)->np.ndarray:
        s = np.random.rand()
        if s > prob:
            return frames_seq
        #h_jitter = np.random.uniform(-10, 10)
        s_jitter = np.random.uniform(-30, 30)  
        v_jitter = np.random.uniform(-30, 30) 
        for i in range(len(frames_seq)):
            hsv = cv2.cvtColor(frames_seq[i], cv2.COLOR_RGB2HSV)
            
            #h = np.mod(hsv[..., 0] + h_jitter,180).astype(np.uint8)
            s = hsv[..., 1] + s_jitter
            v = hsv[..., 2] + v_jitter
            s[s < 0] = 0
            s[s > 254] = 254
            v[v < 0] = 0
            v[v > 254] = 254
            
            #hsv[..., 0] = h
            #hsv[..., 1] = s
            hsv[..., 2] = v
            frames_seq[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        print("Augmentation color jitter: Done!")
        return frames_seq


    
    def __random_brightness(self,brightness_range,frame_seq:np.ndarray)->np.ndarray:
        if len(brightness_range) != 2:
            raise ValueError('`brightness range should be a tuple or list of two floats')
        rand = np.random.uniform(brightness_range[0],brightness_range[1])
        for i in range(np.shape(frame_seq)[0]):
            aug_frame = apply_brightness_shift(frame_seq[i,:,:,:],rand)
            frame_seq[i] = aug_frame
        print(f"Augmentation random brightness: Done!{rand}")
        
        return frame_seq


    def __augment(self,frame_seq):
        if not self._data_augmentation:
            frame_seq = frame_seq * self.rescale

            return frame_seq
        print("\nStarted Augmentation Process")
        frame_seq = self.__random_brightness([0.6,1.4],frame_seq)
        #frame_seq = self.__color_jitter(frame_seq,0.5)
        frame_seq = frame_seq * self.rescale
        frame_seq = self.__horizontal_flip(frame_seq)
        #frame_seq = self.__random_rotation(frame_seq,10,0.7)
        return frame_seq

    def __getOpticalFlow(self,frame_seq):
        """Calculate dense optical flow of input video
        Args:
            video: the input video with shape of [frames,height,width,channel]. dtype=np.array
        Returns:
            flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
            flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
        """
        # initialize the list of optical flows
        gray_video = []
        for i in range(len(frame_seq)):
            img = cv2.cvtColor(frame_seq[i], cv2.COLOR_RGB2GRAY)
            gray_video.append(np.reshape(img,(self.target_size[0],self.target_size[1],1)))

        flows = []
        for i in range(0,len(frame_seq)-1):
            # calculate optical flow between each pair of frames
            flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i+1], None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            # subtract the mean in order to eliminate the movement of camera
            flow[..., 0] -= np.mean(flow[..., 0])
            flow[..., 1] -= np.mean(flow[..., 1])
            # normalize each component in optical flow
            flow[..., 0] = cv2.normalize(flow[..., 0],None,0,255,cv2.NORM_MINMAX)
            flow[..., 1] = cv2.normalize(flow[..., 1],None,0,255,cv2.NORM_MINMAX)
            # Add into list 
            flows.append(flow)
            
        # Padding the last frame as empty array
        flows.append(np.zeros((self.target_size[0],self.target_size[1],2)))
        flows = np.array(flows)
        frames_flows = []
        for frame, flow in zip(frame_seq, flows):
            frames_flows.append(np.append(frame,flow,axis=2))

        frames_flows = np.array(frames_flows, dtype=np.float32)
        return frames_flows

        
    def __data_generation(self,list_IDs_temp):
        X_data = []
        y_data = []
        for _,i in enumerate(list_IDs_temp): #Iterating through each ######################################sequence of frames
            seq_frames = self.data.iloc[i,0]
            y = self.data.iloc[i,1]
            temp_data_list = []
            for img in seq_frames:
                try:
                    # Remove this 
                    img = os.path.join('datasets',img)
                    image = cv2.imread(img)
                    ext_img = self.__transform(image)
                except Exception as e: 
                    '''Code you'd want to run in case of an exception/err'''
                    print(e)
                    (height,width)=self.target_size
                    ext_img = np.zeros((height,width,3), np.uint8)
                temp_data_list.append(ext_img)
            
            temp_data_list = np.array(temp_data_list,dtype=np.float32)
            if self._data_augmentation:
                print(f'Image directory {img}')
                temp_data_list = self.__augment(temp_data_list)
            else:
                temp_data_list = temp_data_list*self.rescale
            if self.append_flows:
                temp_data_list = self.__getOpticalFlow(temp_data_list)
            X_data.append(temp_data_list) 
            y_data.append(y)
        X = np.array(X_data)  #Converting list to array
        

        y = np.array(y_data)
        if self.is_autoencoder == True:
            return X, X
        else:
            return X, np_utils.to_categorical(y,num_classes=self.n_classes)

    def _file_generator(self,data_path,data_files):
        #for every csv(directory) in list of csv files(directories)
        for f in data_files:
            tmp_df = pd.read_csv(os.path.join(data_path,f))
            label_list = list(tmp_df["Label"])
            total_images = len(label_list)

            #Just in case we ask for temporal length bigger than all the images we have
            if total_images >= self.temporal_length:
                num_samples = int((total_images-self.temporal_length)/self.temporal_stride) + 1
                #print('num of samples from vide seq--{}:  {}'.format(f,num_samples))
                img_list = list(tmp_df['FileName'])
            else:
                print('num of frames is less than temporal length; discarding this file--{}'.format(f))
                continue

            start_frame = 0
            #we have a rolling window of size==self.temporal_length and each iteration
            #we add $self.temporal_strides # of elements and pop after the first 3 
            samples = deque()
            samp_count=0
            for img in img_list:
                samples.append(img)
                if len(samples)==self.temporal_length:
                    samples_c= copy.deepcopy(samples)
                    samp_count+=1
                    for t in range(self.temporal_stride):
                        samples.popleft()
                    yield samples_c,label_list[0]



    def _load_samples(self,path='data_files',data_cat='train'):
        data_path = os.path.join(path,data_cat)
        data_files = os.listdir(data_path)
        file_gen = self._file_generator(data_path,data_files)
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