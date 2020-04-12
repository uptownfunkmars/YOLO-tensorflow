import tensorflow as tf
import numpy as np
import cv2 as cv

def generate():
    for i in range(10):
        yield i

num = generate()
num = generate()
print(num.__next__())

# img_path = r'D:\DataSet\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
# img_path += '\\2007_000027.jpg'
# print(img_path)
# img = cv.imread(img_path, cv.COLOR_RGB2GRAY)
# print(img.shape)
# img = cv.resize(img, (480, 640))
#
# cv.namedWindow("temp")
# cv.imshow("temp", img)
# cv.waitKey()
# cv.destroyAllWindows()

# with tf.variable_scope("test"):
#     var1 = tf.get_variable("var1", [5, 480, 640, 3], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
#     var2 = tf.get_variable("var2", [3, 3, 3, 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
#     feature1 = tf.nn.conv2d(var1, var2, strides=[1, 1, 1, 1], padding="SAME")
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
#
# with tf.Session(config=config) as sess:
#     sess.run(tf.global_variables_initializer())
#     # print(sess.run(feature1))
#     # print(sess.run(var1))
#     print(var1.get_shape().as_list())
#     print(sess.run(feature1))
#     print(feature1.get_shape().as_list())
