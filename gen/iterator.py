import numpy as np
from .utils import _load_sequence
import os
import threading

import numpy as np

from keras_preprocessing import get_keras_submodule

try:
    IteratorType = get_keras_submodule('utils').Sequence
except ImportError:
    IteratorType = object


class Iterator(IteratorType):
    """Base class for image data iterators.
    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """
    white_list_formats = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff')

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()
        self.on_epoch_end()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def on_epoch_end(self): 
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            if self.n == 0:
                # Avoiding modulo by zero error
                current_index = 0
            else:
                current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]


    def __len__(self): 
        return (self.n + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)

        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()

        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)



    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self


    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class BatchFromFilesMixin():
    """Adds methods related to getting batches from filenames
    It includes the logic to transform image files to batches.
    """

    def set_processing_attrs(self,
                             frame_seq_generator,
                             target_size,
                             color_mode,
                             data_format,
                             interpolation,
                             keep_aspect_ratio,
                             prefix='datasets'):
        self.prefix = prefix
        self.frame_seq_generator = frame_seq_generator
        self.target_size = tuple(target_size)
        self.keep_aspect_ratio = keep_aspect_ratio
        if color_mode != 'rgb':
            raise NotImplementedError('Invalid color mode:', color_mode,
                             '; expected "rgb"')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                raise NotImplementedError
        else:
                raise NotImplementedError

        self.interpolation = interpolation

    def _get_batches_of_transformed_samples(self, index_array):

        batch_x = []
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self._filepaths
        for i, j in enumerate(index_array):

            #x is a sequence of images
            x = _load_sequence(filepaths[j],
                           target_size=self.image_shape,
                           prefix=self.prefix)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if self.frame_seq_generator:
                x = self.frame_seq_generator.random_augmentation(x)
                if self.append_flows:
                    x = np.float32(x)
                    x = self.frame_seq_generator.append_optical_flow(x)
                elif self.append_diff:
                    x = np.float32(x)
                    x = self.frame_seq_generator.append_frame_differences(x)
                x = self.frame_seq_generator.standardize(x)
            batch_x.append(x)

        batch_x = np.array(batch_x,dtype=self.dtype)
        # optionally save augmented images to disk for debugging purposes
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()

        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]

        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)

            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.

        elif self.class_mode == 'multi_output':
            batch_y = [output[index_array] for output in self.labels]

        elif self.class_mode == 'raw':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        return batch_x, batch_y
        
