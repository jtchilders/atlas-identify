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


class BatchGenerator(Sequence):
   def __init__(self,config,filelist,seed=None,name=''):

      self.config          = config
      self.name            = name

      # get file list
      self.filelist        = filelist
      logger.info('%s: found %s input files',self.name,len(self.filelist))
      if len(self.filelist) < 1:
         raise Exception('%s: length of file list needs to be at least 1' % self.name)
      # logger.error('%s: first file: %s',self.name,self.filelist[0])

      self.shuffle = self.config['data_handling']['shuffle']
      if self.shuffle:
         tmplist = np.array(self.filelist)
         np.random.shuffle(tmplist)
         self.filelist = tmplist.tolist()

      self.n_chan,self.img_h,self.img_w = tuple(config['data_handling']['image_shape'])
   

      self.evts_per_file      = config['data_handling']['evt_per_file']
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

      self.categorical_crossentropy = False
      if self.categorical_crossentropy:
         self.bjet = np.zeros((self.n_classes))
         self.bjet[0] = 1
         self.other = np.zeros((self.n_classes))
         self.other[1] = 1
      


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
         x_batch = np.zeros((self.batch_size, 1, self.n_chan, self.img_h, self.img_w))
         # desired network output
         y_batch = np.zeros((self.batch_size))
         

         ##########
         # calculate which file needed based on list of files, events per file,
         # ad which batch index

         epoch_image_index = batch_index * self.batch_size

         file_index = int(epoch_image_index / self.evts_per_file)
         if file_index >= len(self.filelist):
            raise Exception('{0}: file_index {1} is outside range for filelist {2}'.format(self.name,file_index,len(self.filelist)))
         
         image_index = epoch_image_index % self.evts_per_file
         

         ######
         # open the file
         if self.current_file_index != file_index or self.images is None:
            logger.debug('%s: new file opening %s %s',self.name,self.current_file_index,file_index)
            self.current_file_index = file_index
            logger.debug('%s: opening file: %s',self.name,self.filelist[self.current_file_index])
            file_content = np.load(self.filelist[self.current_file_index])
            self.images = file_content['raw']
            self.truth_classes = file_content['truth']
            # logger.debug('%s: shape images %s truth %s',self.name,self.images.shape,self.truth_classes.shape)
         # else:
         #    logger.debug('%s: not opening file  %s %s',self.name,self.current_file_index,file_index)
         
         logger.error('%s: opening file with idx %s batch_index %s file_index %s image_index %s epoch_image_index %s',self.name,
               idx,
               batch_index,
               file_index,
               image_index,
               epoch_image_index)
            

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

               logger.debug('%s: opening file: %s',self.name,self.filelist[self.current_file_index])
               file_content = np.load(self.filelist[self.current_file_index])
               self.images = file_content['raw']
               self.truth_classes = file_content['truth']
               # logger.debug('%s: shape images %s truth %s',self.name,self.images.shape,self.truth_classes.shape)
               image_index = 0

            # logger.debug('%s: image_index = %s  file_index = %s',self.name,image_index,self.current_file_index)

            # get the image and boxes, must reshape image to include a channel in the case of 3D
            x_batch[i] = np.reshape(self.images[image_index],(1,self.n_chan, self.img_h, self.img_w))
            
            # using categorical_crossentropy
            if self.categorical_crossentropy:
               if self.truth_classes[image_index][0][CLASS_START+3] == 1:
                  y_batch[i] = self.bjet
               else:
                  y_batch[i] = self.other
            else:
               y_batch[i] = self.truth_classes[image_index][0][CLASS_START+3]

            # increase instance counter in current batch
            image_index += 1

         end = time.time()
         average_read_time = (end - start) / self.batch_size

         logger.debug('%s: x_batch = %s',self.name,np.sum(x_batch))
         logger.debug('%s: y_batch = %s',self.name,y_batch)
         logger.debug('%s: x_batch shape = %s',self.name,x_batch.shape)
         # logger.debug('%s: y_batch shape = %s',self.name,y_batch.shape)
         logger.debug('%s: exiting ave read time: %10.4f',self.name,average_read_time)

         # print(' new batch created', idx)

         return x_batch, y_batch
      except Exception as e:
         logger.exception('%s: caught exception %s',self.name,str(e))
         raise

   def on_epoch_end(self):
      if self.shuffle:
         np.random.shuffle(self.filelist)
