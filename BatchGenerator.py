from keras.utils import Sequence
import numpy as np
import logging,multiprocessing,time

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


class BatchGenerator(Sequence):
   def __init__(self,config,filelist,seed=None,name=''):

      self.config          = config
      self.name            = name

      if seed is not None:
         logger.info('setting numpy seed: %s',seed)
         np.random.seed(seed)

      # get file list
      self.filelist        = filelist
      logger.info('%s: found %s input files',self.name,len(self.filelist))
      if len(self.filelist) < 1:
         raise Exception('%s: length of file list needs to be at least 1' % self.name)

      train_file_index = int(config['data_handling']['training_to_validation_ratio'] * len(self.filelist))
      np.random.shuffle(self.filelist)

      self.train_imgs = self.filelist[:train_file_index]
      self.valid_imgs = self.filelist[train_file_index:]

      self.shuffle = self.config['data_handling']['shuffle']

      self.n_chan,self.img_h,self.img_w = tuple(config['data_handling']['image_shape'])
   

      self.evts_per_file      = config['data_handling']['evt_per_file']
      self.nevts              = len(self.filelist) * self.evts_per_file
      self.batch_size         = config['training']['batch_size']
      self.n_classes          = len(config['data_handling']['classes'])

      self.current_file_index = -1
      self.images = None
      self.truth_classes = None


      logger.debug('%s: evts_per_file:           %s',self.name,self.evts_per_file)
      logger.debug('%s: nevts:                   %s',self.name,self.nevts)
      logger.debug('%s: batch_size:              %s',self.name,self.batch_size)
      logger.debug('%s: n_classes:               %s',self.name,self.n_classes)


   def __len__(self):
      return int(float(self.nevts) / self.batch_size)

   def num_classes(self):
      return len(self.config['data_handling']['classes'])

   def size(self):
      return self.nevts

   # return a batch of images starting at the given index
   def __getitem__(self, idx):

      try:
         start = time.time()
         logger.error('%s: starting get batch %s',self.name,idx)

         ##########
         # prepare output variables:

         # input images (1 is there for the 1 channel, in 3D CNN)
         x_batch = np.zeros((self.batch_size, 1, self.n_chan, self.img_h, self.img_w))
         # desired network output
         y_batch = np.zeros((self.batch_size, self.n_classes))
         

         ##########
         # calculate which file needed based on list of files, events per file,
         # ad which batch index

         epoch_image_index = idx * self.batch_size

         file_index = int(epoch_image_index / self.evts_per_file)
         if file_index >= len(self.filelist):
            raise Exception('{0}: file_index {1} is outside range for filelist {2}'.format(self.name,file_index,len(self.filelist)))
         
         image_index = epoch_image_index % self.evts_per_file
         
         logger.debug('%s: opening file with idx %s file_index %s image_index %s',self.name,
               idx,
               file_index,image_index)

         ######
         # open the file
         if self.current_file_index != file_index or self.images is None:
            logger.debug('%s: new file opening %s %s',self.name,self.current_file_index,file_index)
            self.current_file_index = file_index
            file_content = np.load(self.filelist[self.current_file_index])
            self.images = file_content['raw']
            self.truth_classes = file_content['truth'][...,CLASS_START:]
            logger.debug('%s: shape images %s truth %s',self.name,self.images.shape,self.truth_classes.shape)
         else:
            logger.debug('%s: not opening file  %s %s',self.name,self.current_file_index,file_index)
         
            

         ########
         # loop over the batch size
         # create the outputs
         for i in range(self.batch_size):
            logger.debug('%s: loop %s start',self.name,i)

            # if our image index has gone past the number
            # of images per file, then open the next file
            if image_index >= self.evts_per_file:
               logger.debug('%s: new file opening %s',self.name,self.current_file_index)
               self.current_file_index += 1
               
               if self.current_file_index >= len(self.filelist):
                  self.on_epoch_end()
                  self.current_file_index = 0

               file_content = np.load(self.filelist[self.current_file_index])
               self.images = file_content['raw']
               self.truth_classes = file_content['truth'][...,CLASS_START:]
               logger.debug('%s: shape images %s truth %s',self.name,self.images.shape,self.truth_classes.shape)
               image_index = 0

            logger.debug('%s: image_index = %s  file_index = %s',self.name,image_index,self.current_file_index)

            # get the image and boxes, must reshape image to include a channel in the case of 3D
            x_batch[i] = np.reshape(self.images[image_index],(1,self.n_chan, self.img_h, self.img_w))
            y_batch[i] = self.truth_classes[image_index]

            # increase instance counter in current batch
            image_index += 1

         logger.debug('%s: x_batch = %s',self.name,np.sum(x_batch))
         logger.debug('%s: y_batch = %s',self.name,np.sum(y_batch))
         logger.debug('%s: x_batch shape = %s',self.name,x_batch.shape)
         logger.debug('%s: y_batch shape = %s',self.name,y_batch.shape)
         logger.debug('%s: exiting getitem duration: %s, file_index = %s',self.name,(time.time() - start),self.current_file_index)

         # print(' new batch created', idx)

         return x_batch, y_batch
      except Exception as e:
         logger.exception('%s: caught exception %s',self.name,str(e))
         raise

   def on_epoch_end(self):
      if self.shuffle:
         np.random.shuffle(self.filelist)
