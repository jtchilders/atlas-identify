import logging
from keras.layers import Conv2D, Conv3D, Input, MaxPooling2D, MaxPooling3D
from keras.layers import Dense, BatchNormalization, Reshape, Flatten, Dropout
from keras.layers import Activation
from keras.models import Model,Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate

logger = logging.getLogger(__name__)


def build_model(config,print_summary=True):
   
   input_image = Input(shape=tuple(config['data_handling']['image_shape']))
   output = input_image

   layer_num = 0

   # 2 layers with pooling
   for conf in [[128,(3,9),(2,4)],
                [64,(3,3),(2,4)],
                [64,(3,3),(1,2)],
                #[128,(3,3),(1,2)],
                #[128,(3,3),(2,2)],
               ]:
      output = CBLP_layer(output,
               filters=conf[0],
               window=conf[1],
               pool_size=conf[2],
               layer_num = layer_num,
         )
      layer_num += 1

   # layers without pooling
   for conf in [
               [64,(1,1)],
               [128,(3,3)],
               [256,(3,3)],
               ]:
      output = CBL_layer(output,
               filters=conf[0],
               window=conf[1],
               layer_num = layer_num,
         )
      layer_num += 1

   # capture skip connection
   skip_connection = output

   # 4 layers without pooling
   for conf in [
               [512,(3,3)],
               #[256,(1,1)],
               #[512,(3,3)],
               #[1024,(3,3)],
               ]:
      output = CBL_layer(output,
               filters=conf[0],
               window=conf[1],
               layer_num = layer_num,
         )
      layer_num += 1

   output = concatenate([skip_connection,output],axis=1)

   for conf in [[512,(3,3)]]:
      output = CBLP_layer(output,
               filters=conf[0],
               window=conf[1],
               layer_num = layer_num,
         )
      layer_num += 1

   n_grid_boxes_h,n_grid_boxes_w = output.shape[2:4]
   n_grid_boxes_w = int(str(n_grid_boxes_w))
   n_grid_boxes_h = int(str(n_grid_boxes_h))
   config['training']['gridW'] = n_grid_boxes_w
   config['training']['gridH'] = n_grid_boxes_h

   logger.info('grid size: %s x %s',n_grid_boxes_w,n_grid_boxes_h)

   n_classes = len(config['data_handling']['classes'])
   layer_num += 1
   output = Conv2D(4 + 1 + n_classes,
                        (1,1), strides=(1,1),
                        padding='same',
                        name='DetectionLayer_{0}'.format(layer_num),
                        kernel_initializer='lecun_normal',
                        data_format='channels_first')(output)
   output = Reshape((n_grid_boxes_h, n_grid_boxes_w, 4 + 1 + n_classes),name='reshape_{0}'.format(layer_num))(output)


   # boxes = Input(shape=(n_grid_boxes_h, n_grid_boxes_w, 4 + 1 + n_classes))
   # output = Lambda(lambda args: args[0],name='lambda_{0}'.format(layer_num))([output, boxes])

   model = Model(input_image,output)

   if print_summary:
      model.summary()

   return model


def build_model3D(config,args,print_summary=True):
   
   input_image = Input(shape=tuple([1] + config['data_handling']['image_shape']))
   output = input_image

   layer_num = 0

   logger.info('input_image: %s',output.shape)

   # layers with pooling (16, 48, 1152)
   for conf in [['cblp',32,(3,3,10),(1,1,4)],  # (16, 48, 288)
                ['cblp',64,(3,3,3),(1,1,4)],  # (16, 48, 72)
                ['cblp',128,(5,5,5),(2,2,4)],  # (8, 24, 18)
                ['cbl',128,(3,3,3)],  # (8, 24, 18)
                ['cbl',256,(3,3,3)],  # (8, 24, 18)
                ['cblp',512,(3,3,3),(2,2,2)],  # (4, 16, 9)
                ['cbl',256,(2,2,2)],  # (4, 16, 9)
                ['cblp',256,(2,2,2),(1,2,1)],   # (4, 8, 9)
               ]:
      if 'cblp' in conf[0]:
         output = CBLP_layer3D(output,
                  filters=conf[1],
                  window=conf[2],
                  pool_size=conf[3],
                  layer_num = layer_num,
            )
      elif 'cbl' in conf[0]:
         output = CBL_layer3D(output,
                  filters=conf[1],
                  window=conf[2],
                  layer_num = layer_num,
            )
      layer_num += 1


   # flatten the CNN for classification
   output = Flatten(data_format='channels_first',name='flatten_{0}'.format(layer_num))(output)
   layer_num += 1
   
   # dense layer 1
   output = Dense_layer(output,width=128,layer_num=layer_num)
   layer_num += 1

   # dense layer 2
   output = Dense_layer(output,width=128,layer_num=layer_num)
   layer_num += 1

   # dense layer 3
   output = Dense_layer(output,width=128,layer_num=layer_num)
   layer_num += 1

   # class layer
   output = Dense(len(config['data_handling']['classes']),name='dense_{0}'.format(layer_num))(output)
   layer_num += 1
   output = Activation('softmax',name='softmax_{0}'.format(layer_num))(output)

   model = Model(input_image,output)

   if print_summary:
      if args.horovod:
         import horovod.keras as hvd
         if hvd.rank() == 0:
            model.summary()
      else:
         model.summary()

   return model


def build_cifar10_3d(config,args,print_summary=True):
   
   input_image = Input(shape=tuple([1] + config['data_handling']['image_shape']))
   layer_num = 0

   logger.info('input_image: %s',input_image.shape)

   output = Conv3D(128,(4,4,4),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(input_image)
   output = Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = MaxPooling3D(pool_size=(1,1,2), data_format='channels_first',name='pool_{0}'.format(layer_num))(output)
   layer_num += 1
   output = Conv3D(128,(6,6,6),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(output)
   output = Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = MaxPooling3D(pool_size=(1,1,2), data_format='channels_first',name='pool_{0}'.format(layer_num))(output)
   output = Dropout(0.25,name='dropout_{0}'.format(layer_num))(output)
   
   layer_num += 1
   output = Conv3D(64,(6,6,6),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(output)
   output = Activation('relu',name='relu_{0}'.format(layer_num))(output)
   layer_num += 1
   output = Conv3D(64,(6,6,6),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(output)
   output = Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = MaxPooling3D(pool_size=(2,2,4), data_format='channels_first',name='pool_{0}'.format(layer_num))(output)
   output = Dropout(0.25,name='dropout_{0}'.format(layer_num))(output)
   
   layer_num += 1
   output = Conv3D(32,(3,3,3),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(output)
   output = Activation('relu',name='relu_{0}'.format(layer_num))(output)
   layer_num += 1
   output = Conv3D(32,(3,3,3),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(output)
   output = Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = MaxPooling3D(pool_size=(1,2,4), data_format='channels_first',name='pool_{0}'.format(layer_num))(output)
   output = Dropout(0.25,name='dropout_{0}'.format(layer_num))(output)

   # layer_num += 1
   # output = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(output)
   # output = Activation('relu',name='relu_{0}'.format(layer_num))(output)
   # layer_num += 1
   # output = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',name='conv3d_{0}'.format(layer_num),use_bias=False,data_format='channels_first')(output)
   # output = Activation('relu',name='relu_{0}'.format(layer_num))(output)
   # output = MaxPooling3D(pool_size=(2,2,4), data_format='channels_first',name='pool_{0}'.format(layer_num))(output)
   # output = Dropout(0.25,name='dropout_{0}'.format(layer_num))(output)



   layer_num += 1
   output = Flatten(name='flatter_{0}'.format(layer_num))(output)
   output = Dense(512,name='dense_{0}'.format(layer_num))(output)
   output = Activation('relu',name='relu_{0}'.format(layer_num))(output)
   output = Dropout(0.5,name='dropout_{0}'.format(layer_num))(output)
   layer_num += 1
   output = Dense(len(config['data_handling']['classes']),name='dense_{0}'.format(layer_num))(output)
   output = Activation('softmax',name='softmax_{0}'.format(layer_num))(output)

   model = Model(input_image,output)

   if print_summary:
      if args.horovod:
         import horovod.keras as hvd
         if hvd.rank() == 0:
            model.summary()
      else:
         model.summary()

   return model


def CBL_layer(input,
            filters=32,
            window=(3,3),
            strides=(1,1),
            padding='same',
            layer_num=0,
            use_bias=False,
            data_format='channels_first',
            axis=1,
            alpha=0.1,
            ):
   x = Conv2D(filters,window,
            strides=strides,
            padding=padding,
            name='conv2d_{0}'.format(layer_num),
            use_bias=use_bias,
            data_format=data_format)(input)
   x = BatchNormalization(axis=axis,name='norm_{0}'.format(layer_num))(x)
   x = LeakyReLU(alpha=alpha,name='relu_{0}'.format(layer_num))(x)
   return x


def CBLP_layer(input,
            filters=32,
            window=(3,3),
            strides=(1,1),
            padding='same',
            layer_num=0,
            use_bias=False,
            data_format='channels_first',
            axis=1,
            alpha=0.1,
            pool_size=(2,2),
            ):
   x = CBL_layer(input,
            filters=filters,
            window=window,
            strides=strides,
            padding=padding,
            layer_num=layer_num,
            use_bias=use_bias,
            data_format=data_format,
            axis=axis,
            alpha=alpha,
            )
   x = MaxPooling2D(pool_size=pool_size, data_format=data_format,name='pool_{0}'.format(layer_num))(x)
   return x


def CBL_layer3D(input,
            filters=32,
            window=(3,3,3),
            strides=(1,1,1),
            padding='same',
            layer_num=0,
            use_bias=False,
            data_format='channels_first',
            axis=1,
            alpha=0.1,
            ):
   x = Conv3D(filters,window,
            strides=strides,
            padding=padding,
            name='conv3d_{0}'.format(layer_num),
            use_bias=use_bias,
            data_format=data_format)(input)
   x = BatchNormalization(axis=axis,name='norm_{0}'.format(layer_num))(x)
   x = LeakyReLU(alpha=alpha,name='relu_{0}'.format(layer_num))(x)
   return x



def CBLP_layer3D(input,
            filters=32,
            window=(3,3,3),
            strides=(1,1,1),
            padding='same',
            layer_num=0,
            use_bias=False,
            data_format='channels_first',
            axis=1,
            alpha=0.1,
            pool_size=(2,2),
            ):
   x = CBL_layer3D(input,
            filters=filters,
            window=window,
            strides=strides,
            padding=padding,
            layer_num=layer_num,
            use_bias=use_bias,
            data_format=data_format,
            axis=axis,
            alpha=alpha,
            )
   x = MaxPooling3D(pool_size=pool_size, data_format=data_format,name='pool_{0}'.format(layer_num))(x)
   return x


def Dense_layer(input,
                width=4096,
                activation='relu',
                dropout=0.4,
                batch_norm_axis=1,
                layer_num=0):
   output = Dense(width,name='dense_{0}'.format(layer_num))(input)
   output = Activation(activation,name='{0}_{1}'.format(activation,layer_num))(output)
   output = Dropout(dropout,name='dropout_{0}'.format(layer_num))(output)
   output = BatchNormalization(axis=batch_norm_axis,name='norm_{0}'.format(layer_num))(output)
   return output

