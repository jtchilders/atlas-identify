from keras.utils import Sequence
import numpy as np
import logging,time

logger = logging.getLogger(__name__)

# Right now the truth are format as:
# bool objectFound         # if truth particle is in the hard scattering process
# bbox x                   # globe eta
# bbox y                   # globe phi
# bbox width               # Gaussian sigma (required to be 3*sigma<pi)
# bbox height              # Gaussian sigma (required to be 3*sigma<pi)
# bool class1              # truth u/d
# bool class2              # truth s
# bool class3              # truth c
# bool class4              # truth b
# bool class_other         # truth g

# indices in 'truth' vector
BBOX_CENTER_X = 1
BBOX_CENTER_Y = 2
BBOX_WIDTH = 3
BBOX_HEIGHT = 4
CLASS_START = 5
CLASS_END = 10


class SparseBatchGenerator(Sequence):
   def __init__(self,config,filelist,seed=None,name=''):

      self.config          = config
      self.name            = name

      # get file list
      self.filelist        = filelist
      logger.info('%s: found %s input files',self.name,len(self.filelist))
      if len(self.filelist) < 1:
         raise Exception('%s: length of file list needs to be at least 1' % self.name)
      logger.error('%s: first file: %s',self.name,self.filelist[0])

      self.shuffle = self.config['data_handling']['shuffle']

      self.n_chan,self.img_h,self.img_w = tuple(config['data_handling']['image_shape'])
      self.image_shape = tuple(config['data_handling']['image_shape'])
   

      self.evts_per_file      = 1
      self.nevts              = len(self.filelist) * self.evts_per_file
      self.batch_size         = config['training']['batch_size']
      self.n_classes          = len(config['data_handling']['classes'])

      self.current_file_index = -1
      self.images = None
      self.truth_classes = None

      self.rank               = config['rank']
      self.nranks             = config['nranks']
      self.num_batches        = int(np.floor(float(self.nevts) / self.batch_size / self.nranks))


      logger.debug('%s: evts_per_file:           %s',self.name,self.evts_per_file)
      logger.debug('%s: nevts:                   %s',self.name,self.nevts)
      logger.debug('%s: batch_size:              %s',self.name,self.batch_size)
      logger.debug('%s: n_classes:               %s',self.name,self.n_classes)

   def get_event_data(self,file_index):
      file_content = np.load(self.filelist[file_index])
      event = self.importSparse2DenseTensor(file_content[0])
      truth = file_content[2]
      return event,truth


   # Convert a list of scipy sparse arrays in csr format to a 3D sparse tensorflow Tensor
   # sparse_matrices: list of sparse scipy arrays
   def importSparse2DenseTensor(self, sparse_matrices):
      dense_mats = []
      for i, s_mat in enumerate(sparse_matrices):
         dns = np.array(s_mat.todense())
         dense_mats.append(dns)
      return np.stack(dense_mats)  # tf.sparse_to_dense(new_indices, shape, np.concatenate(new_data)).eval()

   def __len__(self):
      return self.num_batches

   def num_classes(self):
      return len(self.config['data_handling']['classes'])

   def size(self):
      return self.nevts

   # return a batch of images starting at the given index
   def __getitem__(self, idx):

      try:
         start = time.time()
         logger.error('%s: starting get batch %s',self.name,idx)

         # convert idx to batch index based on rank ID
         batch_index = self.rank + idx * self.nranks


         ##########
         # prepare output variables:

         # input images (1 is there for the 1 channel, in 3D CNN)
         x_batch = np.zeros((self.batch_size,1) + self.image_shape)
         # desired network output
         y_batch = np.zeros((self.batch_size, self.n_classes))
         

         ##########
         # calculate which file needed based on list of files, events per file,
         # ad which batch index

         file_index = batch_index * self.batch_size

         
         logger.error('%s: opening file with idx %s file_index %s',self.name,
               idx,
               file_index)


         ########
         # loop over the batch size
         # create the outputs
         for i in range(self.batch_size):

            evt,truth = self.get_event_data(file_index + i)

            # get the image and boxes, must reshape image to include a channel in the case of 3D
            x_batch[i,0,...] = np.array(evt)
            y_batch[i] = truth[...,CLASS_START:]

         end = time.time()
         average_read_time = (end - start) / self.batch_size
         
         logger.debug('%s: x_batch = %s',self.name,np.sum(x_batch))
         logger.debug('%s: y_batch = %s',self.name,np.sum(y_batch))
         logger.debug('%s: x_batch shape = %s',self.name,x_batch.shape)
         logger.debug('%s: y_batch shape = %s',self.name,y_batch.shape)
         logger.debug('%s: exiting ave read time: %10.4f, file_index = %s',self.name,average_read_time,self.current_file_index)

         # print(' new batch created', idx)

         return x_batch, y_batch
      except Exception as e:
         logger.exception('%s: caught exception %s',self.name,str(e))
         raise

   def on_epoch_end(self):
      if self.shuffle:
         np.random.shuffle(self.filelist)
