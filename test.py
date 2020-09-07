#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/8/6

import numpy as np
from train import generate_one_batch, yolo_loss, bbox_iou
import tensorflow as tf

if __name__ == '__main__':
    train_path = 'annotation/voc2012_test.txt'
    batch_size = 4
    anchors = np.array([
        [[1.25, 1.625], [2.0, 3.75], [4.125, 2.875]],
        [[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]],
        [[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]
    ])
    num_classes = 20
    max_bbox_per_scale = 30

    # 验证集和训练集
    with open(train_path) as f:
        train_lines = f.readlines()

    a = generate_one_batch(train_lines, batch_size, anchors, num_classes, max_bbox_per_scale, "val")
    image, ls, lm, ll, s, m, l = a.__next__().__getitem__(0)
    conv_lbbox = np.ones((batch_size, 13, 13, 3 * (num_classes + 5)), dtype=np.float32)  # (?, ?, ?, 3*(num_classes+5))
    conv_mbbox = np.ones((batch_size, 26, 26, 3 * (num_classes + 5)), dtype=np.float32)  # (?, ?, ?, 3*(num_classes+5))
    conv_sbbox = np.ones((batch_size, 52, 52, 3 * (num_classes + 5)), dtype=np.float32)  # (?, ?, ?, 3*(num_classes+5))
    label_sbbox = np.ones((batch_size, 52, 52, 3, num_classes + 5), dtype=np.float32) / 2  # (?, ?, ?, 3, num_classes+5)
    label_mbbox = np.ones((batch_size, 26, 26, 3, num_classes + 5), dtype=np.float32) / 2  # (?, ?, ?, 3, num_classes+5)
    label_lbbox = np.ones((batch_size, 13, 13, 3, num_classes + 5), dtype=np.float32) / 2  # (?, ?, ?, 3, num_classes+5)
    true_sbboxes = np.ones((batch_size, max_bbox_per_scale, 4), dtype=np.float32)  # (?, 150, 4)
    true_mbboxes = np.ones((batch_size, max_bbox_per_scale, 4), dtype=np.float32)  # (?, 150, 4)
    true_lbboxes = np.ones((batch_size, max_bbox_per_scale, 4), dtype=np.float32)  # (?, 150, 4)

    # args = [conv_lbbox, conv_mbbox, conv_sbbox, ls.astype(np.float32), lm.astype(np.float32), ll.astype(np.float32), s.astype(np.float32), m.astype(np.float32), l.astype(np.float32)]
    args = [conv_lbbox, conv_mbbox, conv_sbbox, label_sbbox, label_mbbox, label_lbbox,
            true_sbboxes, true_mbboxes, true_lbboxes]
    sess = tf.Session()
    loss = yolo_loss(args, num_classes, 0.7, anchors, 0.5, 0.5, 0.5,sess)
    print(sess.run(loss))

