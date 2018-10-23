from keras import callbacks
from keras import backend as K
import tensorflow as tf

class TB2(callbacks.TensorBoard):

   def on_batch_end(self,batch,logs=None):
      logs.update({'lr': K.eval(self.model.optimizer.lr)})
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
