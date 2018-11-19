import time,logging
from keras import callbacks
from keras import backend as K
import tensorflow as tf
logger = logging.getLogger()

last_time = time.time()


class TB2(callbacks.TensorBoard):
   
   def __init__(self,config,**kwargs):
        self.config = config
        # self.beta_1=self.config['training']['beta_1']
        # self.beta_2=self.config['training']['beta_2']
        super(TB2,self).__init__(**kwargs)

   def on_batch_end(self,batch,logs=None):
      global last_time
      logger.debug('on_batch_end: time %s',time.time() - last_time)
      last_time = time.time()
      lr = self.model.optimizer.lr
      # decay = self.model.optimizer.decay
      # iterations = self.model.optimizer.iterations
      # lrd = lr * (1. / (1. + decay * tf.to_float(iterations)))
      # t = K.cast(iterations, K.floatx()) + 1
      # lr_t = lrd * (K.sqrt(1. - K.pow(self.beta_2, t)) /(1. - K.pow(self.beta_1, t)))
      logs.update({'lr': K.eval(lr)})  # ,'lr_t': K.eval(lr_t),'lrd': K.eval(lrd)})
      super(TB2,self).on_batch_end(batch, logs)



class TB(callbacks.TensorBoard):
   def __init__(self, log_every=1, **kwargs):
      super(TB,self).__init__(**kwargs)
      self.log_every = log_every
      self.counter = 0

   def on_batch_end(self, batch, logs=None):
      self.counter += 1
      if self.counter % self.log_every == 0:
         for name, value in logs.items():
            print('TB: %s = %s' % (name,value))
            if name in ['batch', 'size']:
               continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.counter)
         logs.update({'lr': K.eval(self.model.optimizer.lr)})
         self.writer.flush()
      super(TB,self).on_batch_end(batch, logs)

   def on_epoch_end(self, epoch, logs=None):
      for name, value in logs.items():
         if (name in ['batch', 'size']) or ('val' not in name):
            continue
         summary = tf.Summary()
         summary_value = summary.value.add()
         summary_value.simple_value = value.item()
         summary_value.tag = name
         self.writer.add_summary(summary, epoch)
      logs.update({'lr': K.eval(self.model.optimizer.lr)})
      self.writer.flush()
