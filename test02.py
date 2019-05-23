#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
from lib.utils.timer import Timer
from lib.nets.vgg16 import vgg16
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import  os, sys, cv2
import argparse

CLASSES = ('__background__',
           'mobile_stalls','street_vendor','advertising')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_10000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def vis_detections_video(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    global lastColor, frameRate
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return im

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1] - 20)), (int(bbox[0] + 200), int(bbox[1])), (10, 10, 10), -1)
        cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (int(bbox[0]), int(bbox[1] - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255))  # ,cv2.CV_AA)
        # cv2.imwrite(pic_path + file_name + '_' + str(c) + '.jpg', frame)
        cv2.imwrite(r'D:\googledownload\Faster-RCNN-TensorFlow-Python3.5-master\data\pic/' + 'img_' + str(c) + '.jpg',
                    im)

    return im


def demo(sess,net, im):
    """Detect object classes in an image using pre-computed object proposals."""
    global frameRate
    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    # im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess,net, im)
    timer.toc()
    # print('Detection took {:.3f}s for '
    #       '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    frameRate = 1.0 / timer.total_time
    print("fps: " + str(frameRate))
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections_video(im, cls, dets, thresh=CONF_THRESH)
        cv2.putText(im, '{:s} {:.2f}'.format("FPS:", frameRate), (1750, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        cv2.imshow(video_path.split('/')[len(video_path.split('/')) - 1], im)
        cv2.waitKey(1) #20

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    # tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])
    tfmodel = r'./default/voc_2007_trainval/default\vgg16_faster_rcnn_iter_40000.ckpt'

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    print('\n\nok')

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
    # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 4,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = im_detect(sess,net, im)


    # video_path = r'D:\googledownload\Faster-RCNN-TensorFlow-Python3.5-master\data\video/'
    # videos = os.listdir(video_path)
    # for video_name in videos:
    #     file_name = video_name.split('.')[0]
    #     print(video_path+video_name)
    #     # exit()
    #     folder_name = file_name
    #     os.makedirs(folder_name,exist_ok=True)
    #     vc = cv2.VideoCapture(video_path+video_name)



    video_path = r'data/video/mobile_car.mp4'
    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)  # 获取视频帧速率
    print(fps)
    # success, im = videoCapture.read()
    c = 1
    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False

    timeF = 10 # 视频帧计数间隔频率
    while True:
        success, im = vc.read()
        # print(success)
        # exit()
        if success and (c%timeF == 0):
            demo(sess,net, im)
            # cv2.imwrite(pic_path + file_name + '_' + str(c) + '.jpg', frame)
            # cv2.imwrite(r'D:\googledownload\Faster-RCNN-TensorFlow-Python3.5-master\data\pic/' + 'img_' + str(c) + '.jpg', im)
        c += 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    vc.release()
    cv2.destroyAllWindows()
