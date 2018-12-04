import logging
from keras import layers as kl
from keras.regularizers import l2
from keras.models import Model

logger = logging.getLogger(__name__)


def build_model(config,args,print_summary=True):
   # feature_size = (3,3,3)
   # Pooling = kl.MaxPooling3D
   image_shape = config['data_handling']['image_shape']
   input_image = kl.Input(shape=tuple([1] + image_shape))
   logger.debug('input image = %s',input_image)
   layer_num = 0

   outputs = []
   for i in range(0,image_shape[0],4):

      logger.debug('i = %s',i)
      subimg = kl.Lambda(lambda x: x[:,:,i:i+4,:,:])(input_image)
      
      num_filters = 64
      x = subimg
      x = conv_layer(subimg,num_filters=num_filters)
      # Instantiate the stack of residual units
      for stack in range(2):
         for res_block in range(2):
            strides = (1,1,1)
            if stack > 0 and res_block == 0:  # first layer but not first stack
               strides = (2,2,2)  # downsample
            y = conv_layer(inputs=x,
                          num_filters=num_filters,
                          strides=strides)
            y = conv_layer(inputs=y,
                          num_filters=num_filters,
                          activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
               # linear projection residual shortcut connection to match
               # changed dims
               x = conv_layer(x,
                              num_filters=num_filters,
                              kernel_size=1,
                              strides=strides,
                              activation=None,
                              batch_normalization=False)
            x = kl.add([x, y])
            x = kl.Activation('relu')(x)
         num_filters *= 2
      
      outputs.append(x)

   num_filters = int(num_filters/2)
   logger.debug('filters = %s',num_filters)
   # logger.debug('outputs = %s',outputs)
   x = kl.Concatenate(axis=2)(outputs)
   logger.debug('concat: %s',x)

   # Instantiate the stack of residual units
   for stack in range(2):
      logger.debug('stack: %s',stack)
      for res_block in range(3):
         logger.debug('res_block: %s',res_block)
         strides = (1,1,1)
         if stack > 0 and res_block == 0:  # first layer but not first stack
            strides = (2,2,2)  # downsample
         logger.debug('x: %s',x)
         y = conv_layer(x,
                       num_filters=num_filters,
                       strides=strides)
         logger.debug('y: %s',y)
         y = conv_layer(y,
                       num_filters=num_filters,
                       activation=None)
         logger.debug('y: %s',y)
         if stack > 0 and res_block == 0:  # first layer but not first stack
            # linear projection residual shortcut connection to match
            # changed dims
            x = conv_layer(x,
                           num_filters=num_filters,
                           kernel_size=1,
                           strides=strides,
                           activation=None,
                           batch_normalization=False)
         x = kl.add([x, y])
         x = kl.Activation('relu')(x)
      num_filters *= 2

   logger.debug('out = %s',x)

   x = kl.AveragePooling3D(pool_size=(1,1,2))(x)
   
   y = kl.Flatten()(x)

   # x = kl.Dense(2048,activation='relu',kernel_initializer='normal',name='dense_{0}'.format(layer_num))(output)
   # output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
   # output = kl.Dropout(0.1,name='dropout_{0}'.format(layer_num))(output)
   # layer_num += 1

   outputs = kl.Dense(len(config['data_handling']['classes']),activation='softmax',kernel_initializer='he_normal')(y)
   # output = kl.Activation('softmax',name='softmax_{0}'.format(layer_num))(output)

   model = Model(input_image,outputs)

   line_length = 150
   positions = [.2, .45, .77, 1.]
   if print_summary:
      if args.horovod:
         import horovod.keras as hvd
         if hvd.rank() == 0:
            model.summary(line_length=line_length,positions=positions)
      elif args.ml_comm:
         import ml_comm as mc
         if mc.get_rank() == 0:
            model.summary(line_length=line_length,positions=positions)
      else:
         model.summary(line_length=line_length,positions=positions)

   return model


def conv_layer(inputs,
               num_filters=16,
               kernel_size=(3,3,3),
               strides=(1,1,1),
               activation='relu',
               batch_normalization=True,
               conv_first=True,
               data_format='channels_first'):
   """2D Convolution-Batch Normalization-Activation stack builder
   # Arguments
     inputs (tensor): input tensor from input image or previous layer
     num_filters (int): Conv2D number of filters
     kernel_size (int): Conv2D square kernel dimensions
     strides (int): Conv2D square stride dimensions
     activation (string): activation name
     batch_normalization (bool): whether to include batch normalization
     conv_first (bool): conv-bn-activation (True) or
         bn-activation-conv (False)
   # Returns
     x (tensor): tensor as input to the next layer
   """
   logger.debug('inputs = %s',inputs)
   conv = kl.Conv3D(num_filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4),
               data_format=data_format)

   x = inputs
   if conv_first:
      x = conv(x)
      logger.debug('conv = %s',x)
      if batch_normalization:
         x = kl.BatchNormalization()(x)
      if activation is not None:
         x = kl.Activation(activation)(x)
   else:
      if batch_normalization:
         x = kl.BatchNormalization()(x)
      if activation is not None:
         x = kl.Activation(activation)(x)
      x = conv(x)
   return x

