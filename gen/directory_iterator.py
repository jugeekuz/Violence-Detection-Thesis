import multiprocessing.pool
import os
import numpy as np
from .utils import _load_samples_from_CSV
from .iterator import BatchFromFilesMixin,Iterator
class CSVIterator(BatchFromFilesMixin,Iterator):
    allowed_class_modes = {'binary','sparse','categorical','multi_output','raw', None}

    def __new__(cls,*args,**kwargs):
        try:

            '''
            from tensorflow.keras.utils import Sequence as TFSequence
            if TFSequence not in cls.__bases__:
                cls.__bases__ = cls.__bases__ + (TFSequence,)
            '''
        except ImportError:
            pass
        return super(CSVIterator, cls).__new__(cls)

    def __init__(self,                 
                 csv_directory,
                 frame_generator,
                 temporal_length,                 
                 classes = None,
                 target_size=(224,224),
                 class_mode='binary',
                 batch_size=32,
                 temporal_stride=1,
                 shuffle=True,
                 seed=None,
                 append_flows=False,
                 prefix=None,
                 color_mode='rgb',
                 data_format='channels_last',
                 interpolation='nearest',
                 keep_aspect_ratio=False,
                 dtype='float32'):

        super(CSVIterator, self).set_processing_attrs(frame_generator,
                                                            target_size,
                                                            color_mode,
                                                            data_format,
                                                            interpolation,
                                                            keep_aspect_ratio)

        self.directory = csv_directory
        self.classes = classes
        self.append_flows = append_flows
        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}, expected one of: {}'.format(class_mode,self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype
        #Do stuff here reading async from directory and give to iterator samples and batch size
        self.samples = 0

        #This is redundant since we'll only work with binary for now
        if not classes:
            classes = []
            for subdir in sorted(os.listdir(csv_directory)):
                if os.path.isdir(os.path.join(csv_directory,subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes,range(len(classes))))
            
        #here we should read the csv and read all file directories
        #with one thread per class
        pool = multiprocessing.pool.ThreadPool()
        results = []
        self.filenames = []
        i = 0
        for dirpath in (os.path.join(csv_directory,subdir) for subdir in classes):
            results.append(
                pool.apply_async(_load_samples_from_CSV,
                                 (dirpath,temporal_length,temporal_stride))
            )
        classes_list = []
        for res in results:
            filenames, classes = res.get()
            classes_list += classes
            self.filenames += filenames

        self.samples = len(self.filenames)
        self.classes = np.array(classes_list, dtype='int32')
        '''
        for classes in classes_list:
            self.classes[i:i + len(classes)] = classes
            i += len(classes)
        '''
        print('Found {} images in total, consisting of {} sequences across {} classes'.format( len(classes_list)*temporal_length, len(classes_list), self.num_classes))

        pool.close()
        pool.join()
        self._filepaths = self.filenames
        '''
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        '''
        super(CSVIterator, self).__init__(self.samples,
                                                batch_size,
                                                shuffle,
                                                seed)