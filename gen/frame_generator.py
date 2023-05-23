import warnings
import numpy as np
from multiprocessing import Pool
from functools import partial
from .directory_iterator import CSVIterator
import cv2
from tensorflow.keras.preprocessing.image import apply_brightness_shift, apply_affine_transform
class FrameGenerator(object):
    def __init__(self,
                rescale = 1/255.,
                color_jitter = None,
                brightness_range = None,
                pepper_ratio=100,
                salt_ratio=100,
                rotation_range = 0,                
                data_format='channels_last',
                horizontal_flip = False,
                append_flows = False,
                append_diff = False,
                dtype='float32'):

        self.rescale = rescale
        self.dtype=dtype
        #Check that all values are of correct format

        if not isinstance(horizontal_flip,bool):
            raise ValueError(
                '`horizontal_flip` should of type bool'
                'Received: {}'.format(type(horizontal_flip))
            )
        self.horizontal_flip = horizontal_flip

        if not isinstance(append_flows,bool):
            raise ValueError(
                '`append_flows` should of type bool'
                'Received: {}'.format(type(append_flows))
            )
        self.append_flows = append_flows
        
        if not isinstance(append_diff,bool):
            raise ValueError(
                '`append_diff` should of type bool'
                'Received: {}'.format(type(append_diff))
            )
        self.append_diff = append_diff
        if rotation_range < 0 or rotation_range > 90 or not isinstance(rotation_range,int):
            raise ValueError(
                '`rotation_range` should be int between 0-90 degrees'
                'Received: {}'.format(rotation_range)
            )        
        self.rotation_range = rotation_range

        if color_jitter is not None:
            if (not isinstance(color_jitter, (tuple, list)) or
                    len(color_jitter) != 3):
                raise ValueError(
                    '`color_jitter should be tuple or list of 3 floats. '
                    'Received: %s' % (brightness_range,))
        self.color_jitter = color_jitter

        if not isinstance(pepper_ratio,int):
            raise ValueError(
                '`pepper_ratio` should of type int'
                'Received: {}'.format(type(pepper_ratio))
            )
        self.pepper_ratio = pepper_ratio

        if not isinstance(salt_ratio,int):
            raise ValueError(
                '`salt_ratio` should of type int'
                'Received: {}'.format(type(salt_ratio))
            )
        self.salt_ratio = salt_ratio

        if brightness_range is not None:
            if (not isinstance(brightness_range, (tuple, list)) or
                    len(brightness_range) != 2):
                raise ValueError(
                    '`brightness_range should be tuple or list of two floats. '
                    'Received: %s' % (brightness_range,))
        self.brightness_range = brightness_range

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError(
                '`data_format` should be `"channels_last"` '
                '(channel after row and column) or '
                '`"channels_first"` (channel before row and column). '
                'Received: {}'.format(data_format))

                
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        


    def flow_from_CSV(self,
                      csv_directory,
                      temporal_length,
                      temporal_stride=1,
                      target_size=(224,224),
                      shuffle=True,
                      class_mode='binary',
                      batch_size=32,
                      interpolation='nearest',
                      prefix=None,
                      append_flows = False,
                      append_diff=False,
                      random_seed = None,
                      keep_aspect_ratio=False):

        self.target_size = target_size
        return CSVIterator(
                           csv_directory,
                           self,
                           temporal_length,
                           temporal_stride=temporal_stride,
                           target_size=target_size,
                           keep_aspect_ratio=keep_aspect_ratio,
                           batch_size=batch_size,
                           class_mode=class_mode,
                           shuffle=shuffle,
                           append_flows=append_flows,
                           append_diff=append_diff,
                           prefix=None,
                           seed=random_seed,
                           interpolation=interpolation)


    def standardize(self,x):
        x *= self.rescale
        return x

    def append_optical_flow(self,frame_seq):
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

        frames_flows = np.array(frames_flows, dtype=self.dtype)
        return frames_flows

    def append_frame_differences(self,frame_seq):
        """Calculate frame differences between frames and append        
        """
        # initialize the list of optical flows
        gray_video = []
        for i in range(len(frame_seq)):
            img = cv2.cvtColor(frame_seq[i], cv2.COLOR_RGB2GRAY)
            gray_video.append(np.reshape(img,(self.target_size[0],self.target_size[1],1)))

        differences = []
        for i in range(0,len(frame_seq)-1):
            # calculate frame differences between each pair of frames
            frame_diff = gray_video[i+1] - gray_video[i]
            differences.append(frame_diff)
            
        # Padding the last frame as empty array
        differences.append(np.zeros((self.target_size[0],self.target_size[1],1)))
        #convert to np array
        differences = np.array(differences)

        #append frame differences as need be
        frames_differences = []
        for frame, diff in zip(frame_seq, differences):
            frames_differences.append(np.append(frame,diff,axis=2))

        #convert into np array
        frames_differences = np.array(frames_differences, dtype=self.dtype)
        return frames_differences

    def get_augmentation_paremeters(self, target_size, seed=None):
        
        if seed is not None:
            np.random.seed(seed)

        #color jitter
        if self.color_jitter:
            h_jitter = np.random.uniform(-self.color_jitter[0],self.color_jitter[0])
            s_jitter = np.random.uniform(-self.color_jitter[1],self.color_jitter[1])
            v_jitter = np.random.uniform(-self.color_jitter[2],self.color_jitter[2])
            color_params = [h_jitter,s_jitter,v_jitter]
        else:
            color_params = [0.0, 0.0, 0.0]

        if self.rotation_range:
            theta = np.random.uniform(-self.rotation_range,self.rotation_range)
        else:
            theta = 0

        if self.brightness_range:
            brightness = np.random.uniform(self.brightness_range[0],self.brightness_range[1])
        else:
            brightness = 0

        if self.horizontal_flip:
            flip = np.random.choice([True,False])
        else:
            flip = False
        
        if self.pepper_ratio:
            pepper_ratio = self.pepper_ratio
        else:
            pepper_ratio = 100

        if self.salt_ratio:
            salt_ratio = self.salt_ratio
        else:
            salt_ratio = 100



        augmentation_parameters = {'color_params': color_params,
                                   'theta': theta,
                                   'brightness': brightness,
                                   'flip': flip,
                                   'salt_ratio':salt_ratio,
                                   'pepper_ratio':pepper_ratio}
        return augmentation_parameters


    #x is a frame sequence here not an image
    #I should probably put it in another file to be able to edit the individual transformations easier
    #but its here for now
    def apply_augmentations(self, x, augmentation_parameters):
        #apply transformations here

        #HORIZONTAL FLIP
        '''
        flip = augmentation_parameters.get('flip',False)
        if flip:
            with Pool() as pool:
                args = [(image,1) for image in x]
                results = pool.starmap_async(partial(cv2.flip), args)
                x = np.array(results.get(),dtype=self.dtype)
                #print(f'Successfully pooled type {type(x)} shape {x.shape}')
        '''        
        
        flip = augmentation_parameters.get('flip',False)
        if flip :
            for i in range(len(x)):
                x[i] = cv2.flip(x[i],1)
        

        
        #RANDOM BRIGHTNESS
        brightness = augmentation_parameters.get('brightness',0)
        if brightness != 0:
            for i in range(np.shape(x)[0]):
                x[i] = apply_brightness_shift(x[i,:,:,:],brightness)



        #RANDOM ROTATION
        theta = augmentation_parameters.get('theta',0)
        if theta != 0:
            #hardcoded for now but can change in future
            fill_mode='nearest'
            cval=0.
            interpolation_order=1
            for i in range(np.shape(x)[0]):
                x[i] = apply_affine_transform(x[i, :, :, :], theta=theta,row_axis=self.row_axis-1,col_axis=self.col_axis-1,channel_axis=self.channel_axis-1,
                                        fill_mode=fill_mode, cval=cval,
                                        order=interpolation_order)

        #PEPPER - SETS RANDOM PIXELS TO BLACK
        pepper_ratio = augmentation_parameters.get('pepper_ratio',100)
        if pepper_ratio != 100:
            for i in range(np.shape(x)[0]):
                img_shape = x[i].shape
                noise = np.random.randint(self.pepper_ratio, size=img_shape)
                x[i] = np.where(noise == 0, 0, x[i])


        #SALT - SETS RANDOM PIXELS TO WHITE
        salt_ratio = augmentation_parameters.get('salt_ratio',100)
        if salt_ratio != 100:
            for i in range(np.shape(x)[0]):
                img_shape = x[i].shape
                noise = np.random.randint(self.salt_ratio, size=img_shape)
                x[i] = np.where(noise == 0, 255, x[i])                
        



        return x
    def random_augmentation(self, x, seed=None):
        parameters = self.get_augmentation_paremeters(self.target_size)
        return self.apply_augmentations(x, parameters)
