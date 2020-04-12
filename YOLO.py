import tensorflow as tf
import numpy as np
import cv2 as cv


class YOLO(object):
    def __init__(self):
        self.S = 7
        self.B = 2
        self.C = 20

        self.height = 480
        self.width = 500

        self.alpha = 0.1

        self.x_offset = np.transpose(np.reshape(np.array([np.arange(self.S)] * self.S * self.B), [self.B, self.S, self.S]), [1, 2, 0])
        self.y_offset = np.transpose(self.x_offset, [1, 0, 2])

        self.threshold = 0.2
        self.iou_threshold = 0.4

        self.max_output_size = 10

        self.initializer = tf.contrib.layers.xavier_initializer()

    def get_model(self, x):
        net = self._conv_layer(1, x, 64, 7, 2)
        net = self._leaky_relu(net)
        net = self._maxpool_layer(net, 2, 2)

        net = self._conv_layer(3, net, 192, 3, 1)
        net = self._leaky_relu(net)
        net = self._maxpool_layer(net, 2, 2)

        net = self._conv_layer(4, net, 128, 1, 1)
        net = self._leaky_relu(net)

        net = self._conv_layer(5, net, 256, 3, 1)
        net = self._leaky_relu(net)

        net = self._conv_layer(6, net, 256, 1, 1)
        net = self._leaky_relu(net)

        net = self._conv_layer(7, net, 512, 3, 1)
        net = self._leaky_relu(net)
        net = self._maxpool_layer(net, 2, 2)

        id = 7
        for i in range(4):
            id += 1
            net = self._conv_layer(id, net, 256, 1, 1)
            net = self._leaky_relu(net)
            id += 1
            net = self._conv_layer(id, net, 512, 3, 1)
            net = self._leaky_relu(net)

        id += 1
        net = self._conv_layer(id, net, 512, 1, 1)
        net = self._leaky_relu(net)
        id+= 1
        net = self._conv_layer(id, net, 1024, 3, 1)
        net = self._leaky_relu(net)
        net = self._maxpool_layer(net, 2, 2)

        for i in range(2):
            id += 1
            net = self._conv_layer(id, net, 512, 1, 1)
            net = self._leaky_relu(net)
            id += 1
            net = self._conv_layer(id, net, 1024, 3, 1)
            net = self._leaky_relu(net)

        id += 1
        net = self._conv_layer(id, net, 1024, 3, 1)
        net = self._leaky_relu(net)
        id += 1
        net = self._conv_layer(id, net, 1024, 3, 2)
        net = self._leaky_relu(net)
        id += 1
        net = self._conv_layer(id, net, 1024, 3, 1)
        net = self._leaky_relu(net)
        id += 1
        net = self._conv_layer(id, net, 1024, 3, 1)
        net = self._leaky_relu(net)

        net = self._flatten(net)
        id += 1
        net = self._fc_layer(id, net, 512, self._leaky_relu)
        net = self._fc_layer(id, net, 1024, self._leaky_relu)
        net = tf.nn.dropout(net, 0.5)

        net = self._fc_layer(id, net, self.S * self.S * (self.B * 5 + self.C))
        return net


    def get_tiny_model(self, net_out):
        id = 0
        net = self._conv_layer(id, net_out, 16, 3, 1)
        net = self._maxpool_layer(net, 2, 2)
        id += 1
        net = self._conv_layer(id, net, 32, 3, 1)
        net = self._maxpool_layer(net, 2, 2)
        id += 1
        net = self._conv_layer(id, net, 64, 3, 1)
        net = self._maxpool_layer(net, 2, 2)
        id += 1
        net = self._conv_layer(id, net, 128, 3, 1)
        net = self._maxpool_layer(net, 2, 2)
        id += 1
        net = self._conv_layer(id, net, 256, 2, 2)
        net = self._maxpool_layer(net, 2, 2)
        id += 1
        net = self._conv_layer(id, net, 512, 3, 1)
        net = self._maxpool_layer(net, 2, 2)
        id += 1
        net = self._conv_layer(id, net, 1024, 3, 1)
        id += 1
        net = self._conv_layer(id, net, 1024, 3, 1)
        id += 1
        net = self._conv_layer(id, net, 1024, 3, 1)
        net = self._flatten(net)
        id += 1
        net = self._fc_layer(id, net, 256, self._leaky_relu)
        net = tf.nn.dropout(net, 0.5)
        id += 1
        net = self._fc_layer(id, net, 4096, self._leaky_relu)
        id += 1
        net = self._fc_layer(id, net, 1470)

        return net


    def _conv_layer(self, id, x, num_filters, filter_size, stride):
        with tf.variable_scope("conv_%d" % id):
            in_channels = x.get_shape().as_list()[-1]
            weight = tf.get_variable("weight", [filter_size, filter_size, in_channels, num_filters], dtype=tf.float32,
                                     initializer=self.initializer)
            # weight = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channels, num_filters], stddev=0.1))
            # bias = tf.Variable(tf.zeros([num_filters, ]))

            pad_size = filter_size // 2
            pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
            x_pad = tf.pad(x, pad_mat)
            conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding="VALID")
            output = self._leaky_relu(conv)

            return output

    def _fc_layer(self, id, x, num_out, activation=None):
        with tf.variable_scope("fc_%d" % id):
            num_in = x.get_shape().as_list()[-1]
            weight = tf.get_variable("weight", [num_in, num_out], dtype=tf.float32, initializer=self.initializer)
            # weight = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.1))
            bias = tf.get_variable("bias", [num_out, ], initializer=tf.zeros_initializer())
            # bias = tf.Variable(tf.zeros([num_out, ]))
            output = tf.nn.xw_plus_b(x, weight, bias)
            if activation:
                output = activation(output)

            return output

    def _maxpool_layer(self, x, pool_size, stride):
        output = tf.nn.max_pool(x, [1, pool_size, pool_size, 1], strides=[1, stride, stride, 1], padding="SAME")
        return output

    def _leaky_relu(self, x):
        return tf.maximum(self.alpha * x, x)

    def _flatten(self, x):
        input_shape = x.get_shape().as_list()
        nums = input_shape[1] * input_shape[2] * input_shape[3]
        reshape = tf.reshape(x, [-1, nums])
        return reshape


    def _interpret_output(self, output):
        idx1 = self.S * self.S * self.C
        idx2 = idx1 + self.S * self.S * self.B

        class_probs = tf.reshape(output[0, :idx1], [self.S, self.S, self.C])
        confs = tf.reshape(output[0, idx1:idx2], [self.S, self.S, self.B])
        boxes = tf.reshape(output[0,idx2:], [self.S, self.S, self.B, 4])

        boxes = tf.stack([(boxes[:, :, :, 0] + tf.constant(self.x_offset, dtype=tf.float32)) / self.S * self.width,
                          (boxes[:, :, :, 1] + tf.constant(self.y_offset, dtype=tf.float32)) / self.S * self.height,
                          tf.square(boxes[:, :, :, 2]) * self.width,
                          tf.square(boxes[:, :, :, 3]) * self.height], axis=3)

        scores = tf.expand_dims(confs, -1) * tf.expand_dims(class_probs, 2)

        scores = tf.reshape(scores, [-1, self.C])
        boxes = tf.reshape(boxes, [-1, 4])

        boxes_classes = tf.argmax(scores, axis=11)
        boxes_classes_scores = tf.reduce_max(scores, axis=1)

        filter_mask = boxes_classes_scores >= self.threshold
        scores = tf.boolean_mask(boxes_classes_scores, filter_mask)
        boxes = tf.boolean_mask(boxes, filter_mask)
        box_classes = tf.boolean_mask(boxes_classes, filter_mask)

        _boxes = tf.stack([boxes[:, 0] - 0.5 * boxes[:, 2], boxes[:, 1] - 0.5 * boxes[:, 3],
                           boxes[:, 0] + 0.5 * boxes[:, 2], boxes[:, 1] + 0.5 * boxes[:, 3]])

        nms_indices = tf.image.non_max_suppression(_boxes, scores, self.max_output_size, self.iou_threshold)

        self.scores = tf.gather(scores, nms_indices)
        self.boxes = tf.gather(boxes, nms_indices)
        self.box_classes = tf.gather(box_classes, nms_indices)

    # def _interpret_output_1(self, output):
    #     probs = np.zeros((7, 7, 2, 20))
    #     class_probs = np.reshape(output[0: 980], (7, 7, 20))
    #     scales = np.reshape(output[980:1078], (7, 7, 2))
    #     boxes = np.reshape(output[1078:], (7, 7, 2, 4))
    #     offset = np.transpose(np.reshape(np.array([np.arange(7)] * 14), (2, 7, 7)), (1, 2, 0))
    #
    #     boxes[:, :, :, 0] += offset
    #     boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    #     boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
    #     boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])
    #
    #     for i in range(2):
    #         for j in range(20):
    #             probs[:, :, i, j] = np.multiply(class_probs[:, :j], scales[:, :, i])   # 求解每个框的置信度，为下一步NMS坐准备
    #
    #     filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
    #     filter_mat_boxes = np.nonzero(filter_mat_probs)
    #
    #     boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
    #     probs_filtered = boxes[filter_mat_probs]
    #
    #     clasess_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
    #
    #     argsort = np.array(np.argsort(probs_filtered))[::-1]
    #     boxes_filtered = boxes_filtered[argsort]
    #     probs_filtered = probs_filtered[argsort]
    #
    #     classes_num_filtered = clasess_num_filtered[argsort]
    #
    #     tf.image.non_max_suppression
    #
    #     for i in range(len(boxes_filtered)):
    #         if probs_filtered[i] == 0 : continue
    #         for j in range(i + 1, len(boxes_filtered)):
    #             if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
    #                 probs_filtered[j] = 0.0













