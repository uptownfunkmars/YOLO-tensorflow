import tensorflow as tf
import numpy as np
from YOLO import YOLO
from data import shuffle
from loss_fn import loss_fn


# 可以使用argparse进行改
ann_path = r'D:\DataSet\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations'
img_path = r'D:\DataSet\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
input_h = 480
input_w = 500
input_c = 3

batch_size = 8
# 论文默认 epoch is 135
epoch = 1
S = 7
B = 2
C = 20

labels = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car',
          'motorbike', 'train', 'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor']

DataSet = shuffle(ann_path, img_path, labels, batch_size, epoch)

x = tf.placeholder(tf.float32, shape=[None, input_h, input_w, input_c], name='input')
_probs = tf.placeholder(tf.float32, shape=[None, S * S, C], name='probs')
_confs = tf.placeholder(tf.float32, shape=[None, S * S, B], name='confs')
_coord = tf.placeholder(tf.float32, shape=[None, S * S, B, 4], name='coord')
_proid = tf.placeholder(tf.float32, shape=[None, S * S, C], name='proid')
_areas = tf.placeholder(tf.float32, shape=[None, S * S, B], name='areas')
_upleft = tf.placeholder(tf.float32, shape=[None, S * S, B, 2], name='upleft')
_botright = tf.placeholder(tf.float32, shape=[None, S * S, B, 2], name='botright')

model = YOLO()
output = model.get_tiny_model(x)

print(output.get_shape().as_list())

loss = loss_fn(output, _probs, _confs, _coord, _proid, _areas, _upleft, _botright)
tf.summary.scalar("loss", loss)


'''
原文中的学习率策略为：
    1.Use learning rate equal to 0.01 to train 75 epochs.
    2.Using learning rate equal to 0.001 to train 35 epochs.
    3.Using learning rate equal to 0.0001 to train 30 epochs.
'''

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.01, global_step, 100, 0.95)
train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss)

saver = tf.train.Saver()

merge = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    summary_filter = tf.summary.FileWriter(logdir="./log", graph=sess.graph)

    for epoch in range(135):
        for i in range(2140):
            x_batch, feed_batch = DataSet.__next__()
            summary, loss_value, _ = sess.run([merge, loss, train_op], feed_dict={x: x_batch, _probs: feed_batch['probs'],
                                                                                   _confs: feed_batch['confs'],
                                                                                   _coord: feed_batch['coord'],
                                                                                   _proid: feed_batch['proid'],
                                                                                   _areas: feed_batch['areas'],
                                                                                   _upleft: feed_batch['upleft'],
                                                                                   _botright: feed_batch['botright']})

            print("epoch: %05d, batch: %5d, loss: %.5f" % (epoch, i, loss_value))

        if i % 1000 == 0:
            saver.save(sess, "./checkpoints/model_%04d.ckpt" % (epoch))

        # add validation







