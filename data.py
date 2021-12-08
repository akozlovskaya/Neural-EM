import tensorflow as tf
import h5py
import numpy as np
from parameter import train_size, valid_size, test_size

# GET DATA
class InputPipeLine(object):
    def _open_dataset(self, train_size=train_size, valid_size=valid_size, test_size = test_size):
        # open dataset file
        
        self._hdf5_file = h5py.File(self.filename, 'r')
        self._data_in_file = { data_name: self._hdf5_file[self.usage][data_name][:self.sequence_length + 1] for data_name in self.out_list}
        self.limit = {'training': train_size, 'validation': valid_size, 'test': test_size}[self.usage]
        # fix shapes and datatypes
        self.shapes = {
            data_name: (self.sequence_length, self.batch_size, 1) + self._data_in_file[data_name].shape[-3:]
            for data_name, data in self._data_in_file.items()
        }
        self.shapes['idx'] = ()
        self._dtypes = {data_name: tf.float32 for data_name in self.out_list}
        self._dtypes['idx'] = tf.int32
        
    def __init__(self, usage, batch_size, sequence_length, filename, out_list=('features', 'groups')):
        self.usage = usage
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.batch_num = 0
        self.out_list = out_list
        self.filename = filename
        self._open_dataset()

    def get_n_batches(self):
        return self.limit // self.batch_size
        
    def get_batch(self):
    
        bs = self.batch_size
        i = self.batch_num
        sl = self.sequence_length
        new_batch = { data_name:
            tf.convert_to_tensor(self._data_in_file[data_name][:sl + 1, i*bs:(i+1)*bs][:, :, None], dtype=tf.float32)
            for data_name in self.out_list
        }

        self.batch_num += 1
        return new_batch
    
    def set_zero_batch_state(self):
        self.batch_num = 0
    
    def close(self):
        self._hdf5_file.close()