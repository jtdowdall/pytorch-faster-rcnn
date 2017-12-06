from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import torch

CLASSES = ('__background__',  # always index 0
                'apple',
                'broccoli',
                'banana',
                'cookie',
                'egg_hardboiled',
                'egg_scrambled',
                'egg_sunny_side_up',
                'french_fries',
                'hamburger',
                'hotdog',
                'pasta',
                'pizza',
                'rice',
                'salad',
                'strawberry',
                'tomato')

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return 0

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
                          edgecolor='red', linewidth=3)
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
    return 1

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    classes = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]
        if vis_detections(im, cls, dets, thresh=CONF_THRESH):
            classes.append(cls)
    return classes

cfg.TEST.HAS_RPN = True  # Use RPN for proposals

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--model', required=True)
parser.add_argument('--test', required=True)

args = vars(parser.parse_args())

# test directory
test_dir = args['test'].rstrip('/')

# model path
saved_model = args['model']
gt = {}
for cls in CLASSES:
    gt[cls] = {
    'images' : [], 
    'true_pos' : 0, 
    'true_neg' : 0}
with open(os.path.join(test_dir, 'label.txt')) as f:
    lines = [l.strip().split(' ') for l in f.readlines()]
    for line in lines:
        for cls in line[:1]:
            gt[cls]['images'].append(line[0])

if not os.path.isfile(saved_model):
raise IOError(('{:s} not found.\nDid you download the proper networks from '
               'our server and place them properly?').format(saved_model))

net = resnetv1(num_layers=101)
net.create_architecture(len(classes),
                  tag='default', anchor_scales=[8, 16, 32])

mod = torch.load(saved_model)
net.load_state_dict(mod)

net.eval()
net.cuda()

print('Loaded network {:s}'.format(saved_model))

for im_name in test_dir:
    if not '.jpg' in im_name and not '.JPG' in im_name:
        test_dir.remove(im_name)
        continue
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Demo for data/demo/{}'.format(im_name))
    pred_classes = demo(net, im_name)
    for cls in CLASSES:
        if cls in pred_classes:
            if im_name in gt[cls]['images']:
                gt[cls]['true_pos'] += 1
        else:
            if im_name not in gt[cls]['images']:
                gt[cls]['true_neg'] += 1
    #plt.show()
for cls in CLASSES:
    num_pos = len(gt[cls]['imgs'])
    num_neg = len(test_dir) - num_pos
    det =  0.5 * (gt[cls]['true_pos']/float(num_pos) + gt[cls]['true_neg']/float(num_neg))
    print('{}: {}'.format(cls, det))