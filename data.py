from pascal_preprocess import _pascal_voc_clean_xml
from numpy.random import permutation as perm
from copy import deepcopy
import pickle
import numpy as np
import os
from img_transform import imgCV2_affine_trans, imgCV2_recolor
import cv2 as cv


ann_path = r'D:\DataSet\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations'
img_path = r'D:\DataSet\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
labels = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane',
                     'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
                     'dining table', 'potted plant', 'sofa', 'tv/monitor']

def _fix(obj, dims, scale, offs):
    for i in range(1, 5):
        dim = dims[(i + 1) % 2]
        off = offs[(i + 1) % 2]
        obj[i] = int(obj[i] * scale - off)
        obj[i] = max(min(obj[i], dim), 0)

def resize_input(img, h, w):
    imsz = cv.resize(img, (w, h))
    imsz = imsz / 255.0
    imsz = imsz[:, :, ::-1]
    return imsz

def preprocess(img, resize_h, resize_w, allobj=None):
    if type(img) is not np.ndarray:
        img = cv.imread(img)

    if allobj is not None:
        # 对图片进行放射变换
        result = imgCV2_affine_trans(img)
        img, dims, trans_param = result
        scale, offs, flip = trans_param
        for obj in allobj:
            # 对仿射变换后图片中包含的框与图片一样进行仿射变换。
            _fix(obj, dims, scale, offs)
            # 如果如片进行翻转变换，那么图片中的坐标框也需要进行翻转。
            if not flip: continue
            obj_1_ = obj[1]
            obj[1] = dims[0] - obj[3]
            obj[3] = dims[0] - obj_1_

        # 随机颜色变换。
        img = imgCV2_recolor(img)
    # 调整图片尺寸到输入大小。
    img = resize_input(img, resize_h, resize_w)
    return img


# parse 的作用在与将xml文件处理为list。
def parse(ann, labels):
    if not os.path.isdir(ann):
        msg = 'Annotation directory not found {} .'
        exit('Error: {}'.format(msg.format(ann)))
    print('\n{} parsing {}'.format('YOLO', ann))

    dumps = _pascal_voc_clean_xml(ann, labels)
    return dumps


# 使用batch划分batch时，输入的来源只包含了annotation包含的xml文件。
def batch(chunk, img_path):
    S, B = 7, 2
    C, labels = 20, ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane',
                     'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
                     'dining table', 'potted plant', 'sofa', 'tv/monitor']

    jpg = chunk[0]
    w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    path = img_path + '\\' + jpg
    img = preprocess(path, 480, 500, allobj)

    # Calculate regression target
    # 每个grid尺寸
    cellx = 1.0 * w / S
    celly = 1.0 * h / S

    for obj in allobj:
        # 原始图像中的中心坐标点
        center_x = 0.5 * (obj[1] + obj[3])
        center_y = 0.5 * (obj[2] + obj[4])

        # 除以每个格子的宽度，确定在所在grid的横坐标和纵坐标。
        cx = center_x / cellx
        cy = center_y / celly

        # 如果格子的横纵坐标超出了尺寸，在返回空。
        if cx >= S or cy >= S:
             return None, None

        # 求解每个bound box的宽和高
        obj[3] = float(obj[3] - obj[1]) / w
        obj[4] = float(obj[4] - obj[2]) / h
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        # 确定相对每个grid中的位置。
        obj[1] = cx - np.floor(cx)
        obj[2] = cy - np.floor(cy)
        # 确定在整体中格子grid的编号。
        obj += [int(np.floor(cy) * S + np.floor(cx))]

    probs = np.zeros([S * S, C])
    confs = np.zeros([S * S, B])
    coord = np.zeros([S * S, B, 4])
    proid = np.zeros([S * S, C])
    prear = np.zeros([S * S, 4])

    for obj in allobj:
        # 为该格子的类别全部赋0初始值
        probs[obj[5], :] = [0.0] * C

        #找到GT类别在label中的索引，使用该索引在类别对应位赋为1。
        probs[obj[5], labels.index(obj[0])] = 1.0

        proid[obj[5], :] = [1] * C
        coord[obj[5], :, :] = [obj[1:5]] * B

        # obj[3] ** 2将根式恢复至原始大小，得到的是相对原始图片大小的归一值，乘S后得到相对于特征图的bound box的左上角
        # 和右下角
        prear[obj[5], 0] = obj[1] - obj[3] ** 2 * 0.5 * S
        prear[obj[5], 1] = obj[2] - obj[4] ** 2 * 0.5 * S
        prear[obj[5], 2] = obj[1] + obj[3] ** 2 * 0.5 * S
        prear[obj[5], 3] = obj[2] + obj[4] ** 2 * 0.5 * S

        confs[obj[5], :] = [1.0] * B

    upleft = np.expand_dims(prear[:, 0:2], 1)
    botright = np.expand_dims(prear[:, 2:4], 1)

    wh = botright - upleft
    area = wh[:, :, 0] * wh[:, :, 1]

    upleft = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    inp_feed_val = img
    loss_feed_val = {'probs': probs, 'confs': confs, 'coord': coord, 'proid': proid,
                     'areas': areas, 'upleft': upleft, 'botright': botright}

    return inp_feed_val, loss_feed_val


def shuffle(annotation_path, img_path, labels, batch_size, epoch):
    data = parse(annotation_path, labels)
    size = len(data)

    if batch_size > size:
        batch_size = size

    batch_per_epoch = int(size / batch_size)

    for i in range(epoch):
        shuffle_idx = perm(np.arange(size))
        for b in range(batch_per_epoch):
            x_batch = []
            feed_batch = {}

            for j in range(b * batch_size, b * batch_size + batch_size):
                train_instance = data[shuffle_idx[j]]

                # inp, new_feed = batch(train_instance, img_path)
                try:
                    inp, new_feed = batch(train_instance, img_path)
                except ZeroDivisionError:
                    print("This image's width or height are zeros: ", train_instance[0])
                    print('train_instance: ', train_instance)
                    print('Please remove or fix it then try again.')
                    raise

                if inp is None: continue

                x_batch += [np.expand_dims(inp, 0)]

                for key in new_feed:
                    new = new_feed[key]
                    # python 中的get方法，当该键值存在时返回值，否则返回默认值。
                    old_feed = feed_batch.get(key, np.zeros((0, ) + new.shape))
                    feed_batch[key] = np.concatenate([old_feed, [new]])

            x_batch = np.concatenate(x_batch, 0)

            yield x_batch, feed_batch

        print('Finish {} epoch()es'.format(i + 1))


dataset = shuffle(ann_path, img_path, labels, 8, 1)
x_batch, feed_batch = dataset.__next__()
print(type(feed_batch))
print(feed_batch.keys())

