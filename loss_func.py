from keras import backend as K
from keras import losses
import numpy as np

using_tf = False
if 'tensorflow' in K.backend():
   import tensorflow as tf
   using_tf = True

config = None


def binary_crossentropy(y_true,y_pred):
   loss = losses.binary_crossentropy(y_true,y_pred)
   loss = tf.Print(loss,[loss,y_true,y_pred],'> loss, y_true, y_pred =',-1,1000)
   return loss


def categorical_crossentropy(y_true,y_pred):
   loss = losses.categorical_crossentropy(y_true,y_pred)
   loss = tf.Print(loss,[loss,y_true,y_pred],'> loss, y_true, y_pred =',-1,1000)
   return loss


# learning rate schedule
def step_decay(epoch):
   initial_lrate = 0.01
   drop = 0.5
   epochs_drop = 100.0
   lrate = initial_lrate * np.power(drop, np.floor((1 + epoch) / epochs_drop))
   return lrate


def set_config(local_config):
   global config
   config = local_config


def loss(y_true, y_pred):

   true_sum = K.sum(y_true)
   pred_sum = K.sum(y_pred)

   loss = y_pred - y_true
   loss = K.square(loss)

   if using_tf:
      loss = tf.Print(loss,[tf.timestamp()],'time since epoch',-1,100)
      loss = tf.Print(loss,[true_sum,pred_sum],'sum true/pred = ',-1,100)
      loss = tf.Print(loss,[y_true],'y_true = ',-1,100)

      loss = tf.Print(loss,[y_pred],'y_pred = ',-1,100)

      loss = tf.Print(loss,[loss],'loss = ',-1,100)


   return loss


def loss2(y_true,y_pred):

   # shape (grid_h,grid_w,results vector)
   # results vector = [ 0/1, x, y, w, h, u/d, s, c, b, other]

   tf.sigmoid(y_pred[...,:3])
   tf.sigmoid(y_pred[...,5:])

   y_true    = tf.Print(y_true,[tf.shape(y_true),tf.reduce_sum(y_true)],'y_true shape/reduce_sum = ',-1,100)
   y_pred    = tf.Print(y_pred,[tf.shape(y_pred),tf.reduce_sum(y_pred)],'y_pred shape/reduce_sum = ',-1,100)

   #  (dx + dy) ^ 2
   dxy2 = tf.pow(y_true[...,1:3] - y_pred[...,1:3],2)
   dxy = tf.reduce_sum(dxy2,-1)  # grid_h,grid_w,1

   #  (sqrt(dw) + sqrt(dh) ) ^ 2
   dwh2 = tf.pow(tf.pow(y_true[...,3:5],0.5) - tf.pow(y_pred[...,3:5],0.5),2)
   dwh = tf.reduce_sum(dwh2,-1)  # grid_h,grid_w,1

   # Sum dclass ^ 2
   dclass2 = tf.pow(y_true[...,5:] - y_pred[...,5:],2)
   dclass = tf.reduce_sum(dclass2,-1)  # grid_h,grid_w,1

   # Sum dIOU for grid boxes with predictions
   iou = get_IOU(y_true,y_pred)
   diou = y_true[...,0] * (tf.pow(y_pred[...,0] - iou,2))
   
   loss = dxy + dwh + dclass + diou

   if using_tf:
      loss    = tf.Print(loss,[tf.timestamp()],'time since epoch',-1,100)
      loss    = tf.Print(loss,[tf.shape(dxy2),tf.reduce_sum(dxy2)],'dxy2 shape/reduce_sum = ',-1,100)
      loss    = tf.Print(loss,[tf.shape(dwh2),tf.reduce_sum(dwh2)],'dwh2 shape/reduce_sum = ',-1,100)
      loss    = tf.Print(loss,[tf.shape(dclass2),tf.reduce_sum(dclass2)],'dclass2 shape/reduce_sum = ',-1,100)
      loss    = tf.Print(loss,[tf.shape(iou),tf.reduce_sum(iou)],'iou shape/reduce_sum = ',-1,100)
      loss    = tf.Print(loss,[tf.shape(dxy),tf.reduce_sum(dxy)],'dxy shape/reduce_sum = ',-1,100)
      loss    = tf.Print(loss,[tf.shape(dwh),tf.reduce_sum(dwh)],'dwh shape/reduce_sum = ',-1,100)
      loss    = tf.Print(loss,[tf.shape(dclass),tf.reduce_sum(dclass)],'dclass shape/reduce_sum = ',-1,100)
      loss    = tf.Print(loss,[tf.shape(diou),tf.reduce_sum(diou)],'diou shape/reduce_sum = ',-1,100)
      loss    = tf.Print(loss,[tf.shape(loss),tf.reduce_sum(loss)],'loss shape/reduce_sum = ',-1,100)


   return loss



   


def get_IOU(y_true,y_pred):

   # true box
   true_halfwidth = y_true[...,3] / 2.
   true_halfheight = y_true[...,4] / 2.
   true_x1 = y_true[...,1] - true_halfwidth
   true_y1 = y_true[...,2] - true_halfheight
   true_x2 = y_true[...,1] + true_halfwidth
   true_y2 = y_true[...,2] + true_halfheight

   # pred box
   pred_halfwidth = y_pred[...,3] / 2.
   pred_halfheight = y_pred[...,4] / 2.
   pred_x1 = y_pred[...,1] - pred_halfwidth
   pred_y1 = y_pred[...,2] - pred_halfheight
   pred_x2 = y_pred[...,1] + pred_halfwidth
   pred_y2 = y_pred[...,2] + pred_halfheight


   xA = tf.maximum(pred_x1,true_x1)
   yA = tf.maximum(pred_y1,true_y1)
   xB = tf.minimum(pred_x2,true_x2)
   yB = tf.minimum(pred_y2,true_y2)

   intersect = (tf.maximum(xB - xA,tf.zeros(tf.shape(xB))) *
                tf.maximum(yB - yA,tf.zeros(tf.shape(yB))))

   trueArea = y_true[...,3] * y_true[...,4]
   predArea = y_pred[...,3] * y_pred[...,4]

   union = trueArea + predArea - intersect

   return tf.div_no_nan(intersect,union)


