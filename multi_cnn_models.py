import logging
from keras import layers as kl
from keras.models import Model

logger = logging.getLogger(__name__)


def build_model(config,args,print_summary=True):
   
   image_shape = config['data_handling']['image_shape']
   input_image = kl.Input(shape=tuple([1] + image_shape))
   logger.debug('input image = %s',input_image)
   layer_num = 0

   outputs = []
   for i in range(0,image_shape[0],4):

      logger.debug('i = %s',i)
      output = kl.Lambda(lambda x: x[:,:,i:i+4,:,:],name='lambda_{0}'.format(layer_num))(input_image)
      
      output = kl.Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(output)
      output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
      output = kl.MaxPooling3D(pool_size=(2,2,2), data_format='channels_first',name='pool_{0}'.format(layer_num))(output)
      output = kl.Dropout(0.25,name='dropout_{0}'.format(layer_num))(output)
      layer_num += 1

      output = kl.Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(output)
      output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
      output = kl.MaxPooling3D(pool_size=(2,2,2), data_format='channels_first',name='pool_{0}'.format(layer_num))(output)
      output = kl.Dropout(0.3,name='dropout_{0}'.format(layer_num))(output)
      layer_num += 1

      output = kl.Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(output)
      output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
      output = kl.MaxPooling3D(pool_size=(1,2,2), data_format='channels_first',name='pool_{0}'.format(layer_num))(output)
      output = kl.Dropout(0.3,name='dropout_{0}'.format(layer_num))(output)
      layer_num += 1

      outputs.append(output)

   logger.debug('outputs = %s',outputs)
   output = kl.Concatenate(axis=2,name='concat_{0}'.format(layer_num))(outputs)


   output = kl.Conv3D(512,(3,3,3),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(output)
   output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = kl.MaxPooling3D(pool_size=(2,2,2), data_format='channels_first',name='pool_{0}'.format(layer_num))(output)
   output = kl.Dropout(0.25,name='dropout_{0}'.format(layer_num))(output)
   layer_num += 1

   output = kl.Conv3D(512,(3,3,3),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(output)
   output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = kl.MaxPooling3D(pool_size=(1,2,2), data_format='channels_first',name='pool_{0}'.format(layer_num))(output)
   output = kl.Dropout(0.25,name='dropout_{0}'.format(layer_num))(output)
   layer_num += 1
   
   output = kl.Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(output)
   output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = kl.MaxPooling3D(pool_size=(1,1,2), data_format='channels_first',name='pool_{0}'.format(layer_num))(output)
   output = kl.Dropout(0.3,name='dropout_{0}'.format(layer_num))(output)
   layer_num += 1

   output = kl.Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(output)
   output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = kl.MaxPooling3D(pool_size=(1,1,2), data_format='channels_first',name='pool_{0}'.format(layer_num))(output)
   output = kl.Dropout(0.25,name='dropout_{0}'.format(layer_num))(output)
   layer_num += 1
   
   output = kl.Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(output)
   output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = kl.MaxPooling3D(pool_size=(1,1,2), data_format='channels_first',name='pool_{0}'.format(layer_num))(output)
   output = kl.Dropout(0.25,name='dropout_{0}'.format(layer_num))(output)
   layer_num += 1
   
   output = kl.Flatten(name='flatter_{0}'.format(layer_num))(output)
   output = kl.Dense(256,name='dense_{0}'.format(layer_num))(output)
   output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = kl.Dropout(0.2,name='dropout_{0}'.format(layer_num))(output)
   layer_num += 1

   output = kl.Dense(len(config['data_handling']['classes']),name='dense_{0}'.format(layer_num))(output)
   output = kl.Activation('softmax',name='softmax_{0}'.format(layer_num))(output)

   model = Model(input_image,output)

   if print_summary:
      if args.horovod:
         import horovod.keras as hvd
         if hvd.rank() == 0:
            model.summary()
      else:
         model.summary()

   return model

