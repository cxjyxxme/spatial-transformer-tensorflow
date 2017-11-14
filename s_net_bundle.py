# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import tensorflow as tf
from spatial_transformer2 import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import cv2
from resnet import output_layer
import get_data
from config import *
import time
from tensorflow.contrib.slim.nets import resnet_v2
slim = tf.contrib.slim

def get_4_pts(theta, batch_size):
    with tf.name_scope('get_4_pts'):
        pts1_ = []
        pts2_ = []
        p = []
        temp = []
        for i in range(grid_h):
            p.append([])
            for j in range(grid_w):
                pn = tf.slice(theta, [0, i, j, 0], [-1, 1, 1, -1])
                pn = tf.reshape(pn, [batch_size, 2, 4])
                p[i].append(pn)

        h = 2.0 / grid_h
        w = 2.0 / grid_w
        
        for i in range(grid_h):
            for j in range(grid_w):
                cnt = 0
                this_p = []
                this_p.append([tf.slice(p[i][j], [0, 0, 0], [-1, -1, 1])])
                this_p.append([tf.slice(p[i][j], [0, 0, 1], [-1, -1, 1])])
                this_p.append([tf.slice(p[i][j], [0, 0, 2], [-1, -1, 1])])
                this_p.append([tf.slice(p[i][j], [0, 0, 3], [-1, -1, 1])])
                if (i > 0):
                    this_p[0].append(tf.slice(p[i - 1][j], [0, 0, 2], [-1, -1, 1]))
                    this_p[1].append(tf.slice(p[i - 1][j], [0, 0, 3], [-1, -1, 1]))
                if (j > 0):
                    this_p[0].append(tf.slice(p[i][j - 1], [0, 0, 1], [-1, -1, 1]))
                    this_p[2].append(tf.slice(p[i][j - 1], [0, 0, 3], [-1, -1, 1]))
                if (i < grid_h - 1):
                    this_p[2].append(tf.slice(p[i + 1][j], [0, 0, 0], [-1, -1, 1]))
                    this_p[3].append(tf.slice(p[i + 1][j], [0, 0, 1], [-1, -1, 1]))
                if (j < grid_w - 1):
                    this_p[1].append(tf.slice(p[i][j + 1], [0, 0, 0], [-1, -1, 1]))
                    this_p[3].append(tf.slice(p[i][j + 1], [0, 0, 2], [-1, -1, 1]))
                p0_ = tf.constant([-1 + (j) * w, -1 + (i) * h], shape=[2, 1], dtype=tf.float32)
                p1_ = tf.constant([-1 + (j + 1) * w, -1 + (i) * h], shape=[2, 1], dtype=tf.float32)
                p2_ = tf.constant([-1 + (j) * w, -1 + (i + 1) * h], shape=[2, 1], dtype=tf.float32)
                p3_ = tf.constant([-1 + (j + 1) * w, -1 + (i + 1) * h], shape=[2, 1], dtype=tf.float32)
                p0 = tf.add_n(this_p[0]) / len(this_p[0]) + p0_
                p1 = tf.add_n(this_p[1]) / len(this_p[1]) + p1_
                p2 = tf.add_n(this_p[2]) / len(this_p[2]) + p2_
                p3 = tf.add_n(this_p[3]) / len(this_p[3]) + p3_
                g = tf.concat([p0, p1, p2, p3], 2)
                pts1_.append(tf.reshape(g, [batch_size, 1, 8]))
                pts2_.append(tf.reshape(p0, [batch_size, 1, 2]))
                if (j == grid_w - 1):
                    pts2_.append(tf.reshape(p1, [batch_size, 1, 2]))
                if (i == grid_h - 1):
                    temp.append(tf.reshape(p2, [batch_size, 1, 2]))
                if ((i == grid_h - 1) and (j == grid_w - 1)):
                    temp.append(tf.reshape(p3, [batch_size, 1, 2]))

        pts2_.extend(temp)
        pts1 = tf.reshape(tf.concat(pts1_, 1), [batch_size, grid_h, grid_w, 8])
        pts2 = tf.reshape(tf.concat(pts2_, 1), [batch_size, grid_h + 1, grid_w + 1, 2])
    return pts1, pts2
'''
    theta = tf.reshape(theta, [-1, 3, 3])
    h = 1.0 / grid_h
    w = 1.0 / grid_w
    grid = tf.constant([-w, w, -w, w, -h, -h, h, h, 1, 1, 1, 1], shape=[12], dtype=tf.float32)
    grid = tf.tile(grid, [batch_size * grid_h * grid_w])
    grid = tf.reshape(grid, [-1, 3, 4])

    T_g = tf.matmul(theta, grid)
    x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
    z_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])
    
    t_1 = tf.constant(1.0, shape=[batch_size * grid_h * grid_w, 1, 4])
    t_0 = tf.constant(0.0, shape=[batch_size * grid_h * grid_w, 1, 4])      

    sign_z = tf.where(tf.greater(z_s, t_0), t_1, t_0) * 2.0 - 1.0
    z_s = z_s + sign_z * 1e-8
    x_s = tf.div(x_s, z_s)  #batch_size * grid_h * grid_w, 1, 4
    y_s = tf.div(y_s, z_s)

    dh_s_ = []
    dw_s_ = []
    dh_s = []
    dw_s = []
    for i in range(grid_h):
        for j in range(grid_w):
            dh = i - grid_h / 2.0 + 0.5
            dw = j - grid_w + 0.5
            dh_s_.extend([dh, dh, dh, dh])
            dw_s_.extend([dw, dw, dw, dw])
    for i in range(batch_size):
        dh_s.extend(dh_s_)
        dw_s.extend(dw_s_)

    dx_s = tf.constant(dw_s, shape=[batch_size * grid_h * grid_w, 1, 4], dtype=tf.float32)
    dy_s = tf.constant(dh_s, shape=[batch_size * grid_h * grid_w, 1, 4], dtype=tf.float32)
    x_s = x_s + dx_s
    y_s = y_s + dy_s

    output = tf.concat([x_s, y_s], 1)
    return tf.reshape(output, [batch_size, grid_h, grid_w, 8])
'''

def get_black_pos(pts):
    with tf.name_scope('black_pos'):
        one_ = tf.ones([batch_size, grid_h, grid_w, 8])
        zero_ = tf.zeros([batch_size, grid_h, grid_w, 8])
        black_err = tf.where(tf.greater(pts, one_), pts - one_, zero_) + tf.where(tf.greater(one_ * -1, pts), one_ * -1 - pts, zero_)
    return tf.reshape(black_err, [batch_size, -1])

def calc_distortion_loss(p0, p1, p2, clock, hw):
    h = 2.0 / grid_h
    w = 2.0 / grid_w
    if (hw == 0):
        k = h / w
    else:
        k = w / h

    if (not clock):
        R_ = [0, -k, k, 0]
    else:
        R_ = [0, k, -k, 0]
    R = tf.constant(R_, shape=[4], dtype=tf.float32)
    R = tf.tile(R, [batch_size * grid_h * grid_w])
    R = tf.reshape(R, [-1, 2, 2])
    loss = tf.abs(tf.matmul(R, p1 - p0) - (p2 - p1))    #batch_size*grid_h*grid_w, 2, 1
    return loss * loss

def get_distortion_loss(pts):
    with tf.name_scope('distortion_loss'):
        pts = tf.reshape(pts, [-1, 2, 4])
        p0 = tf.slice(pts, [0, 0, 0], [-1, -1, 1])
        p1 = tf.slice(pts, [0, 0, 1], [-1, -1, 1])
        p2 = tf.slice(pts, [0, 0, 2], [-1, -1, 1])
        p3 = tf.slice(pts, [0, 0, 3], [-1, -1, 1])
        loss =          calc_distortion_loss(p0, p1, p3, 0, 0)
        loss = loss +   calc_distortion_loss(p1, p3, p2, 0, 1)
        loss = loss +   calc_distortion_loss(p3, p2, p0, 0, 0)
        loss = loss +   calc_distortion_loss(p2, p0, p1, 0, 1)
        loss = loss +   calc_distortion_loss(p1, p0, p2, 1, 0)
        loss = loss +   calc_distortion_loss(p0, p2, p3, 1, 1)
        loss = loss +   calc_distortion_loss(p2, p3, p1, 1, 0)
        loss = loss +   calc_distortion_loss(p3, p1, p0, 1, 1)
    return tf.reduce_mean(loss) / 8

def get_consistency_loss(pts):
    with tf.name_scope('consistency_loss'):
        p = []
        for i in range(grid_h + 1):
            p.append([])
            for j in range(grid_w + 1):
                pn = tf.slice(pts, [0, i, j, 0], [-1, 1, 1, -1])
                pn = tf.reshape(pn, [batch_size, 2, 1])
                p[i].append(pn)

        errs = []
        for i in range(grid_h + 1):
            for j in range(grid_w + 1):
                if (i > 1):
                    errs.append(tf.abs(2 * p[i - 1][j] - p[i][j] - p[i - 2][j]))
                if (j > 1):
                    errs.append(tf.abs(2 * p[i][j - 1] - p[i][j] - p[i][j - 2]))
                if (i < grid_h - 1):
                    errs.append(tf.abs(2 * p[i + 1][j] - p[i][j] - p[i + 2][j]))
                if (j < grid_w - 1):
                    errs.append(tf.abs(2 * p[i][j + 1] - p[i][j] - p[i][j + 2]))
        loss = tf.concat(errs, 2)
        loss = tf.reduce_mean(loss * loss)
    return loss
'''
def get_neighbor_loss(pts):
    with tf.name_scope('neighbor_loss'):
        cnt = 0
        losses = []
        for i in range(grid_h):
            for j in range(grid_w):
                p = tf.slice(pts, [0, i, j, 0], [-1, 1, 1, -1])
                p = tf.reshape(p, [batch_size, 2, 4])
                if (i > 0):
                    pn = tf.slice(pts, [0, i - 1, j, 0], [-1, 1, 1, -1])
                    pn = tf.reshape(pn, [batch_size, 2, 4])
                    losses.append(tf.abs(tf.slice(p, [0, 0, 0], [-1, -1, 1]) - tf.slice(pn, [0, 0, 2], [-1, -1, 1])))
                    losses.append(tf.abs(tf.slice(p, [0, 0, 1], [-1, -1, 1]) - tf.slice(pn, [0, 0, 3], [-1, -1, 1])))
                if (j > 0):
                    pn = tf.slice(pts, [0, i, j - 1, 0], [-1, 1, 1, -1])
                    pn = tf.reshape(pn, [batch_size, 2, 4])
                    losses.append(tf.abs(tf.slice(p, [0, 0, 0], [-1, -1, 1]) - tf.slice(pn, [0, 0, 1], [-1, -1, 1])))
                    losses.append(tf.abs(tf.slice(p, [0, 0, 2], [-1, -1, 1]) - tf.slice(pn, [0, 0, 3], [-1, -1, 1])))
    loss = tf.concat(losses, 0)
    loss = tf.reduce_mean(loss * loss)
    return loss


def reduce_layer(input):
    with tf.variable_scope('reduce_layer'):
        with tf.variable_scope('conv0'):
            conv0_ = conv_bn_relu_layer(input, [1, 1, 2048, 512], 1)

        with tf.variable_scope('conv1_0'):
            conv1_0_ = conv_bn_relu_layer2(conv0_, [1, 16, 512, 512], [1, 1])
        with tf.variable_scope('conv2_0'):
            conv2_0_ = conv_bn_relu_layer2(conv1_0_, [9, 1, 512, 512], [1, 1])

        with tf.variable_scope('conv1_1'):
            conv1_1_ = conv_bn_relu_layer2(conv0_, [9, 1, 512, 512], [1, 1])
        with tf.variable_scope('conv2_1'):
            conv2_1_ = conv_bn_relu_layer2(conv1_1_, [1, 16, 512, 512], [1, 1])
        with tf.variable_scope('conv2'):
            conv2_ = conv2_0_ + conv2_1_
        with tf.variable_scope('conv3'):
            conv3_ = conv_bn_relu_layer(conv2_, [1, 1, 512, 128], 1)
        with tf.variable_scope('conv4'):
            conv4_ = conv_bn_relu_layer(conv3_, [1, 1, 128, 32], 1)
        with tf.variable_scope('fc'):
            out = output_layer(tf.reshape(conv4_, [batch_size, 32]), 8)
    return out
'''
def get_resnet(x_tensor, reuse, is_training, x_batch_size):
    with tf.variable_scope('resnet', reuse=reuse):
        dict = resnet_v2.resnet_arg_scope()
        dict['is_training'] = is_training
        dict['reuse'] = reuse
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            resnet, end_points = resnet_v2.resnet_v2_50(x_tensor, global_pool=False, output_stride=32)
            with tf.variable_scope('end_conv'):
                resnet = slim.conv2d(resnet, 256, 3, stride=1, scope='conv1')
                resnet = slim.conv2d(resnet, 64, 3, stride=1, scope='conv2')
                resnet = slim.conv2d(resnet, 8, 1, stride=1, scope='conv3',
                                normalizer_fn=None,
                                activation_fn=None)
       
            theta = resnet
            id_loss = tf.reduce_mean(tf.abs(theta)) * id_mul
    return theta, id_loss

def inference_stable_net(reuse):
    with tf.variable_scope('stable_net'):
        with tf.name_scope('input'):
            # %% Since x is currently [batch, height*width], we need to reshape to a
            # 4-D tensor to use it in a convolutional graph.  If one component of
            # `shape` is the special value -1, the size of that dimension is
            # computed so that the total size remains constant.  Since we haven't
            # defined the batch dimension's shape yet, we use -1 to denote this
            # dimension should not change size.
            x_tensor = tf.placeholder(tf.float32, [None, height, width, tot_ch], name = 'x_tensor')
            x_batch_size = tf.shape(x_tensor)[0]
            x = tf.slice(x_tensor, [0, 0, 0, before_ch], [-1, -1, -1, 1])
            
            for i in range(tot_ch):
                temp = tf.slice(x_tensor, [0, 0, 0, i], [-1, -1, -1, 1])
                tf.summary.image('x' + str(i), temp)

        with tf.name_scope('label'):
            y = tf.placeholder(tf.float32, [None, height, width, 1])
            x4 = tf.slice(y, [0, 0, 0, 0], [-1, -1, -1, 1])
            tf.summary.image('label', x4)

        theta, id_loss = get_resnet(x_tensor, reuse = reuse, is_training=True, x_batch_size = x_batch_size)
        theta_infer, id_loss_infer = get_resnet(x_tensor, reuse = True, is_training=False, x_batch_size = x_batch_size)

        pts1, pts2 = get_4_pts(theta, x_batch_size)
        _, pts2_infer = get_4_pts(theta_infer, x_batch_size)
        with tf.name_scope('inference'):
            h_trans_infer, black_pix_infer = transformer(x, pts2_infer)
        with tf.name_scope('theta_loss'):
            use_theta_loss = tf.placeholder(tf.float32)
            theta_loss = id_loss #theta_loss * use_theta_loss + id_loss

        with tf.name_scope('black_loss'):
            use_black_loss = tf.placeholder(tf.float32)
            black_pos = get_black_pos(pts1)
            black_pos = black_pos * black_pos
            black_pos = black_pos * use_black_loss
            black_pos_loss = tf.reduce_mean(black_pos)

        with tf.name_scope('regu_loss'):
            #regu_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            #regu_loss = tf.add_n(regu_loss)
            regu_loss = tf.add_n(slim.losses.get_regularization_losses())

        distortion_loss = get_distortion_loss(pts1)
        consistency_loss = get_consistency_loss(pts2)
        #neighbor_loss = get_neighbor_loss(pts)

        h_trans, black_pix = transformer(x, pts2)
        tf.summary.image('output', h_trans)
        tf.add_to_collection('output', h_trans)
        with tf.name_scope('img_loss'):
            black_pix = tf.reshape(black_pix, [batch_size, height, width, 1]) 
            #black_pix = tf.stop_gradient(black_pix)
            img_err = (h_trans - y) * (1 - black_pix)
            tf.summary.image('err', img_err * img_err)
            img_loss = tf.reduce_sum(tf.reduce_sum(img_err * img_err, [1, 2, 3]) / (tf.reduce_sum((1 - black_pix), [1, 2, 3]) + 1e-8), [0]) / batch_size

        use_theta_only = tf.placeholder(tf.float32)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([tf.group(*update_ops)]):
            total_loss = theta_loss * theta_mul + ((1 - use_theta_only) * 
            (img_loss * img_mul + regu_loss * regu_mul + black_pos_loss * black_mul + distortion_loss * distortion_mul + consistency_loss * consistency_mul))
        
    ret = {}
    ret['error'] = tf.abs(h_trans - y)
    ret['black_pos'] = black_pos
    ret['black_pix'] = black_pix
    ret['theta_loss'] = theta_loss * theta_mul
    ret['black_loss'] = black_pos_loss * black_mul
    ret['distortion_loss'] = distortion_loss * distortion_mul
    ret['consistency_loss'] = consistency_loss * consistency_mul
    ret['img_loss'] = img_loss * img_mul
    ret['regu_loss'] = regu_loss * regu_mul
    ret['x_tensor'] = x_tensor
    ret['use_theta_only'] = use_theta_only
    ret['y'] = y
    ret['output'] = h_trans
    ret['total_loss'] = total_loss
    ret['use_theta_loss'] = use_theta_loss
    ret['use_black_loss'] = use_black_loss
    return ret
