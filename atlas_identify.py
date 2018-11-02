#!/usr/bin/env python
import os,argparse,logging,json,glob,datetime
import numpy as np
import models
from BatchGenerator import BatchGenerator
from SparseBatchGenerator import SparseBatchGenerator
import loss_func

import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import backend as keras_backend
from callbacks import TB2

import tensorflow as tf
print('done importing')
logger = logging.getLogger(__name__)



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
   parser.add_argument('--lr', default=0.01, type=int,
                       help='learning rate')
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
   parser.add_argument('--timeline_filename',default='timeline_profile.json',
                       help='filename to use for timeline json data')
   parser.add_argument('--sparse', action='store_true',
                       help="Indicate that the input data is in sparse format")
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


   logger.info('keras from:            %s',keras.__file__)
   logger.info('keras version:         %s',keras.__version__)
   logger.info('tensorflow from:       %s',tf.__file__)
   logger.info('config_file:           %s',args.config_file)
   logger.info('tb_logdir:             %s',args.tb_logdir)
   logger.info('horovod:               %s',args.horovod)
   logger.info('num_files:             %s',args.num_files)
   logger.info('lr:                    %s',args.lr)
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
   if args.horovod:
      learning_rate = config_file['training']['learning_rate'] * hvd.size()
   else:
      learning_rate = config_file['training']['learning_rate']
   logger.info('learning_rate:         %s',learning_rate)

   if args.timeline:
      from tensorflow.python.client import timeline
   
   config_proto = create_config_proto(args)
   keras_backend.set_session(tf.Session(config=config_proto))

   # build model
   model = models.build_model3D(config_file,args,print_summary=(rank == 0))

   # get inputs
   train_gen,valid_gen = get_image_generators(config_file,args)

   logger.info('train_gen:             %s',len(train_gen))
   logger.info('valid_gen:             %s',len(valid_gen))

   # pass configuration to loss function
   loss_func.set_config(config_file)

   # create optmization function

   logger.debug('create Adam')
   optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

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
      model.compile(loss='categorical_crossentropy', optimizer=optimizer)

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

   # add stepwise learning rate
   # lrate = LearningRateScheduler(loss_func.step_decay)
   # callbacks.append(lrate)
   

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
         '''
         tensorboard = TB(log_dir=log_path,
                        histogram_freq=config_file['tensorboard']['histogram_freq'],
                        write_graph=config_file['tensorboard']['write_graph'],
                        write_images=config_file['tensorboard']['write_images'],
                        write_grads=config_file['tensorboard']['write_grads'],
                        embeddings_freq=config_file['tensorboard']['embeddings_freq'])
         '''
         tensorboard = TB2(log_dir=log_path,update_freq='batch')
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
          tensorboard = TB2(log_dir=log_path,update_freq='batch')
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

      checkpoint = ModelCheckpoint(config_file['model_pars']['model_checkpoint_file'].format(date=dateString),
                     monitor='val_loss',
                     verbose=1,
                     save_best_only=True,
                     mode='min',
                     period=1)
      callbacks.append(checkpoint)

      # create tensorboard callback
      # create tensorboard callback
      '''
      tensorboard = TB(log_dir=log_path,
                     histogram_freq=config_file['tensorboard']['histogram_freq'],
                     write_graph=config_file['tensorboard']['write_graph'],
                     write_images=config_file['tensorboard']['write_images'],
                     write_grads=config_file['tensorboard']['write_grads'],
                     embeddings_freq=config_file['tensorboard']['embeddings_freq'])
      '''
      tensorboard = TB2(log_dir=log_path,update_freq='batch')
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

   train_file_index = int(config_file['data_handling']['training_to_validation_ratio'] * nfiles)
   np.random.shuffle(filelist)

   Generator = BatchGenerator
   if config_file['sparse']:
      Generator = SparseBatchGenerator

   train_imgs = filelist[:train_file_index]
   train_gen = Generator(config_file,train_imgs,name='train')
   valid_imgs = filelist[train_file_index:nfiles]
   valid_gen = Generator(config_file,valid_imgs,name='valid')

   logger.info(' %s training batches; %s validation batches',len(train_gen),len(valid_gen))

   return train_gen,valid_gen


if __name__ == "__main__":
   print('start main')
   main()
