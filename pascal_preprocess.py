import xml.etree.ElementTree as ET
import os
import sys
import glob

def _pp(l):
    for i in l : print('{}: {}'.format(i, l[i]))

def _pascal_voc_clean_xml(ANN, pick, exclusive = False):
    print("Parsing for {} {}".format(pick, 'exclusive' * int(exclusive)))

    dumps = []
    cur_dir = os.listdir('.')
    os.chdir(ANN)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations) + '*.xml')
    size = len(annotations)

    for i, file in enumerate(annotations):
        in_file = open(file)

        tree = ET.parse(in_file)
        root = tree.getroot()

        jpg = str(root.find('filename').text)
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
        all = []

        for obj in root.iter('object'):
            current = []
            name = obj.find('name').text

            if name not in pick:
                continue

            xmlbox = obj.find('bndbox')
            xn = int(float(xmlbox.find('xmin').text))
            xx = int(float(xmlbox.find('xmax').text))
            yn = int(float(xmlbox.find('ymin').text))
            yx = int(float(xmlbox.find('ymax').text))

            current = [name, xn, yn, xx, yx]
            all += [current]

        add = [[jpg, [w, h, all]]]
        dumps += add
        in_file.close()


    stat = {}
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in pick:
                if current[0] in stat:
                    stat[current[0]] += 1
                else:
                    stat[current[0]] = 1

    _pp(stat)
    return dumps


# ANN = r'D:\DataSet\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations'
# pick = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
#         'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
#         'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor']
#
# dumps = _pascal_voc_clean_xml(ANN, pick)
# print(len(dumps))
#


