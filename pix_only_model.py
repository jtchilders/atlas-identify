import logging
import keras.layers as kl
from keras.models import Model

logger = logging.getLogger(__name__)


def build_model(config,print_summary=True):
   Pooling3D = MaxPooling3D
   input_image = kl.Input(shape=tuple([1] + config['data_handling']['image_shape']))
   logger.debug('input image = %s',input_image)
   output = input_image

   # get only pixel hits
   output = kl.Lambda(lambda x: x[:,:,0:4,:,:])(output)
   logger.debug('output = %s',output)


   output = Conv3D(64,(3,6,18))(output)
   output = kl.BatchNormalization()(output)
   output = kl.Activation('relu')(output)

   output = Pooling3D((2,2,2))(output)
   output = kl.Dropout(0.2)(output)

   output = Conv3D(128,(3,3,3))(output)
   output = kl.BatchNormalization()(output)
   output = kl.Activation('relu')(output)

   output = Pooling3D((1,2,2))(output)
   output = kl.Dropout(0.2)(output)

   output = Conv3D(256,(3,3,3))(output)
   output = kl.BatchNormalization()(output)
   output = kl.Activation('relu')(output)

   output = Pooling3D((1,2,2))(output)
   output = kl.Dropout(0.2)(output)

   output = Conv3D(512,(3,3,3))(output)
   output = kl.BatchNormalization()(output)
   output = kl.Activation('relu')(output)

   output = Pooling3D((1,2,2))(output)

   output = Conv3D(1024,(3,3,3))(output)
   output = kl.BatchNormalization()(output)
   output = kl.Activation('relu')(output)

   output = Pooling3D((1,2,2))(output)





   output = kl.Flatten()(output)

   output = kl.Dense(len(config['data_handling']['classes']),activation=None,kernel_initializer='he_normal')(output)
   output = kl.Activation('sigmoid')(output)

   model = Model(input_image,output)

   # print summary
   line_length = 150
   positions = [.4, .7, .77, 1.]
   if print_summary:
      model.summary(line_length=line_length,positions=positions)

   return model



def AvePooling3D(pool_size=(2, 2, 2),
                 strides=None,
                 padding='valid',
                 data_format='channels_first'):
   return kl.AveragePooling3D(pool_size,
               strides=strides,
               padding=padding,
               data_format=data_format)


def MaxPooling3D(pool_size=(2, 2, 2),
                 strides=None,
                 padding='valid',
                 data_format='channels_first'):
   return kl.MaxPooling3D(pool_size,
               strides=strides,
               padding=padding,
               data_format=data_format)

def Conv3D(filters,
           kernel_size = (3,3,3),
           strides=(1, 1, 1),
           padding='same',
           data_format='channels_first',
           dilation_rate=(1, 1, 1),
           activation=None,
           use_bias=True,
           kernel_initializer='glorot_uniform',
           bias_initializer='zeros',
           kernel_regularizer=None,
           bias_regularizer=None,
           activity_regularizer=None,
           kernel_constraint=None,
           bias_constraint=None,
          ):


   return kl.Conv3D(filters = filters,
                    kernel_size = kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint,
                  )

