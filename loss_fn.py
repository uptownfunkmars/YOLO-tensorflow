import tensorflow as tf
import pickle
import numpy as np
import os

sprob = 1.0
sconf = 5.0
snoob = 1.0
scoor = 1.0

S, B, C = 7, 2, 20
SS = S * S


def loss_fn(net_out, _probs, _confs, _coord, _proid, _areas, _upleft, _botright):
    size1 = [None, SS, C]
    size2 = [None, SS, B]

    coords = net_out[:, SS * (C + B):]
    coords = tf.reshape(coords, [-1, SS, B, 4])
    wh = tf.pow(coords[:, :, :, 2:4], 2)
    area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
    centers = coords[:, :, :, 0:2]
    floor = centers - (wh * 0.5)
    ceil = centers + (wh * 0.5)

    intersect_upleft = tf.maximum(floor, _upleft)
    intersect_botright = tf.minimum(ceil, _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

    iou = tf.truediv(intersect,  _areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.multiply(best_box, _confs)

    conid = snoob * (1.0 - confs) + sconf * confs
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    proid = sprob * _proid

    probs = tf.layers.flatten(_probs)
    proid = tf.layers.flatten(proid)
    confs = tf.layers.flatten(confs)
    conid = tf.layers.flatten(conid)
    coord = tf.layers.flatten(_coord)
    cooid = tf.layers.flatten(cooid)

    true = tf.concat([probs, confs, coord], 1)
    wght = tf.concat([proid, conid, cooid], 1)
    loss = tf.pow(net_out - true, 2)
    loss = tf.multiply(loss, wght)
    loss = tf.reduce_sum(loss, 1)
    loss = 0.5 * tf.reduce_mean(loss)
    return loss

