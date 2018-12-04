#!/usr/bin/env python
import os,argparse,logging,json,glob,datetime
import numpy as np
import models,nn_models,multi_cnn_models,pix_only_model
from BatchGenerator import BatchGenerator
from SparseBatchGenerator import SparseBatchGenerator
import loss_func

import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler
from keras import backend as keras_backend
from callbacks import TB2

import tensorflow as tf
print('done importing')
logger = logging.getLogger(__name__)

rates = []
def lrsched(epoch):
   try:
      return rates[epoch]
   except KeyError:
      return rates[-1]



def create_config_proto(params):
   '''EJ: TF config setup'''
   config = tf.ConfigProto()
   config.intra_op_parallelism_threads = params.num_intra
   config.inter_op_parallelism_threads = params.num_inter
   config.allow_soft_placement         = True
   os.environ['KMP_BLOCKTIME'] = str(params.kmp_blocktime)
   os.environ['KMP_AFFINITY'] = params.kmp_affinity
   return config


def main():
   global rates
   ''' simple starter program that can be copied for use when starting a new script. '''
   parser = argparse.ArgumentParser(description='Atlas Training')
   parser.add_argument('--config_file', '-c',
                       help='configuration in standard json format.')
   parser.add_argument('--tb_logdir', '-l',
                       help='tensorboard logdir for this job.',default=None)
   parser.add_argument('--horovod', default=False,
                       help='use Horovod',action='store_true')
   parser.add_argument('--ml_comm', default=False,
                       help='use Cray PE ML Plugin',action='store_true')
   parser.add_argument('--num_files','-n', default=-1, type=int,
                       help='limit the number of files to process. default is all')
   parser.add_argument('--num_intra', type=int,default=4,
                       help='num_intra')
   parser.add_argument('--num_inter', type=int,default=1,
                       help='num_inter')
   parser.add_argument('--kmp_blocktime', type=int, default=10,
                       help='KMP BLOCKTIME')
   parser.add_argument('--kmp_affinity', default='granularity=fine,compact,1,0',
                       help='KMP AFFINITY')
   parser.add_argument('--batch_queue_size',type=int,default=4,
                       help='number of batch queues in the fit_generator')
   parser.add_argument('--batch_queue_workers',type=int,default=0,
                       help='number of batch workers in the fit_generator')
   parser.add_argument('--timeline',action='store_true',default=False,
                       help='enable chrome timeline profiling')
   parser.add_argument('--adam',action='store_true',default=False,
                       help='use adam optimizer')
   parser.add_argument('--sgd',action='store_true',default=False,
                       help='use SGD optimizer')
   parser.add_argument('--lrsched',action='store_true',default=False,
                       help='use learning rate scheduler')
   parser.add_argument('--timeline_filename',default='timeline_profile.json',
                       help='filename to use for timeline json data')
   parser.add_argument('--sparse', action='store_true',
                       help="Indicate that the input data is in sparse format")
   parser.add_argument('--write_grads', action='store_true',
                       help="Add gradient histograms to tensorboard")
   parser.add_argument('--random-seed', type=int,default=0,dest='random_seed',
                       help="Set the random number seed. This needs to be the same for all ranks to ensure the batch generator serves data properly.")

   args = parser.parse_args()

   print('loading MPI bits')
   log_level = logging.DEBUG
   rank = 0
   nranks = 1
   if args.horovod:
      import horovod.keras as hvd
      hvd.init()
      rank = hvd.rank()
      nranks = hvd.size()
      if rank > 0:
         log_level = logging.ERROR
      logging.basicConfig(level=log_level,format='%(asctime)s %(levelname)s:' + '{:05d}'.format(rank) + ':%(name)s:%(process)s:%(thread)s:%(message)s')
      logger.info('horovod from: %s',hvd.__file__)
      logger.info("Rank: %s of %s",rank,nranks)
   if args.ml_comm:
      logger.debug("importing ml_comm")
      import ml_comm as mc
      from plugin_keras import InitPluginCallback, BroadcastVariablesCallback, DistributedOptimizer
      mc.init_mpi()
      rank = mc.get_rank()
      nranks = mc.get_nranks()
      if rank > 0:
         log_level = logging.ERROR
      logging.basicConfig(level=log_level,format='%(asctime)s %(levelname)s:' + '{:05d}'.format(rank) + ':%(name)s:%(process)s:%(thread)s:%(message)s')
      logger.info('ml_comm from: %s',mc.__file__)
      logger.info("Rank: %s of %s",rank,nranks)
   else:
      logging.basicConfig(level=log_level,format='%(asctime)s %(levelname)s:%(name)s:%(process)s:%(thread)s:%(message)s')
      logger.info('no MPI run')


   logger.info('keras from:            %s',keras.__file__)
   logger.info('keras version:         %s',keras.__version__)
   logger.info('tensorflow from:       %s',tf.__file__)
   logger.info('config_file:           %s',args.config_file)
   logger.info('tb_logdir:             %s',args.tb_logdir)
   logger.info('horovod:               %s',args.horovod)
   logger.info('adam:                  %s',args.adam)
   logger.info('sgd:                   %s',args.sgd)
   logger.info('lrsched:               %s',args.lrsched)
   logger.info('num_files:             %s',args.num_files)
   logger.info('num_intra:             %s',args.num_intra)
   logger.info('kmp_blocktime:         %s',args.kmp_blocktime)
   logger.info('kmp_affinity:          %s',args.kmp_affinity)
   logger.info('batch_queue_size:      %s',args.batch_queue_size)
   logger.info('batch_queue_workers:   %s',args.batch_queue_workers)
   logger.info('timeline:              %s',args.timeline)
   logger.info('timeline_filename:     %s',args.timeline_filename)
   logger.info('random-seed:           %s',args.random_seed)
   logger.info('sparse:                %s',args.sparse)
   np.random.seed(args.random_seed)

   # load configuration
   config_file = json.load(open(args.config_file))
   batch_size = config_file['training']['batch_size']
   num_epochs = config_file['training']['epochs']
   config_file['rank'] = rank
   config_file['nranks'] = nranks
   config_file['sparse'] = args.sparse

   # set learning rate
   logger.info('learning_rate:         %s',config_file['training']['learning_rate'])
   logger.info('beta_1:                %s',config_file['training']['beta_1'])
   logger.info('beta_2:                %s',config_file['training']['beta_2'])
   logger.info('epsilon:               %s',config_file['training']['epsilon'])
   logger.info('decay:                 %s',config_file['training']['decay'])

   if args.timeline:
      from tensorflow.python.client import timeline
   
   config_proto = create_config_proto(args)
   keras_backend.set_session(tf.Session(config=config_proto))

   # build model
   # model = multi_cnn_models.build_model(config_file,args,print_summary=(rank == 0))
   model = pix_only_model.build_model(config_file,print_summary=(rank == 0))

   # get inputs
   train_gen,valid_gen = get_image_generators(config_file,args)

   logger.info('train_gen:             %s',len(train_gen))
   logger.info('valid_gen:             %s',len(valid_gen))

   if len(train_gen) <= 0:
      logger.error('no batches in train generator')
      raise Exception('no batches in train generator')
   if len(valid_gen) <= 0:
      logger.error('no batches in valid generator')
      raise Exception('no batches in valid generator')


   # pass configuration to loss function
   loss_func.set_config(config_file)

   # create optmization function

   logger.debug('create optimizer')
   
   optimizer = optimizers.rmsprop(lr=config_file['training']['learning_rate'],
                        decay=config_file['training']['decay'])
   
   if args.adam:
      optimizer = optimizers.Adam(lr=config_file['training']['learning_rate'],
                        beta_1=config_file['training']['beta_1'],
                        beta_2=config_file['training']['beta_2'],
                        epsilon=config_file['training']['epsilon'],
                        decay=config_file['training']['decay'])
   elif args.sgd:
      optimizer = optimizers.SGD(lr=config_file['training']['learning_rate'],
         momentum=0.0, decay=config_file['training']['decay'], nesterov=False)

   logger.info('optimizer: %s',str(optimizer))
   

   if args.horovod:
      logger.debug('create horovod optimizer')
      optimizer = hvd.DistributedOptimizer(optimizer)
   elif args.ml_comm:
      logger.debug("Distributed Cray optimizer")
      optimizer = DistributedOptimizer(optimizer)

   logger.debug('compile model')
   if args.timeline:
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      model.compile(loss=loss_func.loss2, optimizer=optimizer, options=run_options, run_metadata=run_metadata)
   else:
      model.compile(loss=loss_func.binary_crossentropy, optimizer=optimizer,
                     metrics=[binary_sum_squares,ave_pred,predaccuracy])


   # try to add gradient histograms
   if args.write_grads:
      for layer in model.layers:
         for weight in layer.weights:
            mapped_weight_name = weight.name.replace(':', '_')
            tf.summary.histogram(mapped_weight_name, weight)
            grads = model.optimizer.get_gradients(model.total_loss,weight)

            def is_indexed_slices(grad):
                return type(grad).__name__ == 'IndexedSlices'
            grads = [
                grad.values if is_indexed_slices(grad) else grad
                for grad in grads
               ]
            tf.summary.histogram('{}_grad'.format(mapped_weight_name),
                                                grads)


   # To use Cray Plugin we need to calculate the number of trainable variables
   # Also useful to adjust number of epochs run
   if args.ml_comm:
      trainable_count = int(
        np.sum([keras_backend.count_params(p) for p in set(model.trainable_weights)]))
      # non_trainable_count = int(
      #   np.sum([keras_backend.count_params(p) for p in set(self.model.non_trainable_weights)]))

      # Adjust number of Epochs based on number of ranks used
      # nb_epochs = int(nb_epochs / self.mc.get_nranks())
      if num_epochs == 0:
        num_epochs = 1
      total_steps = int(np.ceil(num_epochs * len(train_gen)))

      # if hvd.rank() == 0:
      # if mc.get_rank() == 0:
      #  print('Total params: {:,}'.format(trainable_count + non_trainable_count))
      #  print('Trainable params: {:,}'.format(trainable_count))
      #  print('Non-trainable params: {:,}'.format(non_trainable_count))
      #  print('Calculation of total_steps: {:,}'.format(total_steps))
   
   # create checkpoint callback
   dateString = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d-%H-%M-%S')
   
   # create log path for tensorboard
   log_path = config_file['tensorboard']['log_dir'] + '_' + dateString
   if args.tb_logdir is not None:
      log_path = args.tb_logdir + '_' + dateString
   logger.info('tensorboard logdir: %s',log_path)
   

   callbacks = []

   if args.lrsched:
      rates = config_file['training']['lrsched']
      callbacks.append(LearningRateScheduler(lrsched,verbose=1))

   verbose = config_file['training']['verbose']
   if args.horovod:
      
      # Horovod: broadcast initial variable states from rank 0 to all other processes.
      # This is necessary to ensure consistent initialization of all workers when
      # training is started with random weights or restored from a checkpoint.
      callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
      
      # Horovod: average metrics among workers at the end of every epoch.
      #
      # Note: This callback must be in the list before the ReduceLROnPlateau,
      # TensorBoard or other metrics-based callbacks.
      callbacks.append(hvd.callbacks.MetricAverageCallback())

      
      

      if rank == 0:
         verbose = config_file['training']['verbose']
         os.makedirs(log_path)

         checkpoint = ModelCheckpoint(config_file['model_pars']['model_checkpoint_file'].format(date=dateString),
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        mode='min',
                        period=1)
         callbacks.append(checkpoint)

         # create tensorboard callback
         tensorboard = TB2(config_file,log_dir=log_path,update_freq='batch')
         callbacks.append(tensorboard)

         
      else:
         verbose = 0
   elif args.ml_comm:
      logger.debug("cray-plugin callbacks")
      callbacks = []
      # Cray ML Plugin: broadcast callback
      init_plugin = InitPluginCallback(total_steps, trainable_count)
      callbacks.append(init_plugin)
      broadcast   = BroadcastVariablesCallback(0)
      callbacks.append(broadcast)

      if rank == 0:
          verbose = config_file['training']['verbose']
          os.makedirs(log_path)

          # create tensorboard callback
          tensorboard = TB2(config_file,log_dir=log_path,update_freq='batch')
          callbacks.append(tensorboard)

          checkpoint = ModelCheckpoint(config_file['model_pars']['model_checkpoint_file'].format(date=dateString),
                        monitor='loss',
                        verbose=1,
                        save_best_only=True,
                        mode='min',
                        period=1)
          callbacks.append(checkpoint)
      else:
          verbose = 0
   else:
      os.makedirs(log_path)
      verbose = config_file['training']['verbose']
      checkpoint = ModelCheckpoint(config_file['model_pars']['model_checkpoint_file'].format(date=dateString),
                     monitor='loss',
                     verbose=1,
                     save_best_only=True,
                     mode='min',
                     period=1)
      callbacks.append(checkpoint)

      # create tensorboard callback
      # create tensorboard callback
      
      tensorboard = TB2(config_file,log_dir=log_path,update_freq='batch')
      callbacks.append(tensorboard)


   logger.debug('callbacks: %s',callbacks)


   logger.debug('call fit generator')
   model.fit_generator(generator         = train_gen,
                        epochs           = config_file['training']['epochs'],
                        verbose          = verbose,
                        validation_data  = valid_gen,
                        callbacks        = callbacks,
                        workers          = args.batch_queue_workers,
                        max_queue_size   = args.batch_queue_size,
                        steps_per_epoch  = len(train_gen),
                        validation_steps = config_file['training']['steps_per_valid'],
                        use_multiprocessing=False)
   logger.debug('done fit gen')


   if args.timeline:
      logger.info('output timeline profile')
      trace = timeline.Timeline(step_stats=run_metadata.step_stats)
      with open(args.timeline_filename, 'w') as f:
         f.write(trace.generate_chrome_trace_format())

   logger.info('done')
   

def get_image_generators(config_file,args):
   # get file list
   filelist = sorted(glob.glob(config_file['data_handling']['input_file_glob']))
   logger.info('found %s input files',len(filelist))
   if len(filelist) < 2:
      raise Exception('length of file list needs to be at least 2 to have train & validate samples')

   nfiles = len(filelist)
   if args.num_files > 0:
      nfiles = args.num_files

   nbatches = int(nfiles*config_file['data_handling']['evt_per_file'] / config_file['training']['batch_size'])
   
   train_file_index = int(config_file['data_handling']['training_to_validation_ratio'] * nfiles)
   np.random.shuffle(filelist)

   Generator = BatchGenerator
   if config_file['sparse']:
      Generator = SparseBatchGenerator

   train_imgs = filelist[:train_file_index]
   valid_imgs = filelist[train_file_index:nfiles]
   logger.info('training index: %s',train_file_index)
   while len(valid_imgs) * config_file['data_handling']['evt_per_file'] / config_file['training']['batch_size'] < 1.:
      logger.info('training index: %s',train_file_index)
      train_file_index -= 1
      train_imgs = filelist[:train_file_index]
      valid_imgs = filelist[train_file_index:nfiles]
         

   train_gen = Generator(config_file,train_imgs,name='train')
   valid_gen = Generator(config_file,valid_imgs,name='valid')

   logger.info(' %s training batches; %s validation batches',len(train_gen),len(valid_gen))

   return train_gen,valid_gen

# for binary case
def predaccuracy(y_true,y_pred):
   y_pred_bool = y_pred > 0.7
   y_true_bool = y_true > 0.7
   correct = tf.to_float(tf.logical_and(y_pred_bool,y_true_bool))
   predac = tf.reduce_sum(correct) / tf.to_float(tf.shape(y_true)[0])
   predac = tf.Print(predac,[predac,y_true,y_pred],'predac,y_true,y_pred = ',-1,1000)
   return predac

def ave_pred(y_true,y_pred):
   ave = tf.reduce_sum(y_pred) / tf.to_float(tf.shape(y_pred)[0])
   return ave

# for binary case
def binary_sum_squares(y_true,y_pred):
   diff_sum = tf.reduce_sum(tf.square(y_true - y_pred)) / tf.to_float(tf.shape(y_true)[0])
   diff_sum = tf.Print(diff_sum,[diff_sum,y_true,y_pred],'binary_sum_sq,y_true,y_pred = ',-1,1000)
   return diff_sum

if __name__ == "__main__":
   print('start main')
   main()
