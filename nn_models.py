import logging
from keras import layers as kl
from keras.models import Model

logger = logging.getLogger(__name__)


def build_model(config,args,print_summary=True):
   
   image_shape = config['data_handling']['image_shape']
   pixels = 1
   for i in image_shape:
      pixels *= i
   input_image = kl.Input(shape=tuple([1] + image_shape))
   output = input_image

   layer_num = 0
   pixels = pixels / 500
   output = kl.Flatten(name='flatter_{0}'.format(layer_num))(input_image)
   output = kl.Dense(int(pixels),name='dense_{0}'.format(layer_num))(output)
   output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = kl.Dropout(0.2,name='dropout_{0}'.format(layer_num))(output)

   layer_num += 1
   pixels = pixels * 2
   output = kl.Dense(int(pixels),name='dense_{0}'.format(layer_num))(output)
   output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = kl.Dropout(0.3,name='dropout_{0}'.format(layer_num))(output)

   layer_num += 1
   pixels = pixels * 2
   output = kl.Dense(int(pixels),name='dense_{0}'.format(layer_num))(output)
   output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = kl.Dropout(0.4,name='dropout_{0}'.format(layer_num))(output)

   layer_num += 1
   pixels = pixels/2
   output = kl.Dense(int(pixels),name='dense_{0}'.format(layer_num))(output)
   output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = kl.Dropout(0.5,name='dropout_{0}'.format(layer_num))(output)

   layer_num += 1
   pixels = pixels/2
   output = kl.Dense(int(pixels),name='dense_{0}'.format(layer_num))(output)
   output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = kl.Dropout(0.4,name='dropout_{0}'.format(layer_num))(output)

   layer_num += 1
   pixels = pixels/2
   output = kl.Dense(int(pixels),name='dense_{0}'.format(layer_num))(output)
   output = kl.Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = kl.Dropout(0.3,name='dropout_{0}'.format(layer_num))(output)

   layer_num += 1
   pixels = pixels/2
   output = kl.Dense(int(pixels),name='dense_{0}'.format(layer_num))(output)
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

