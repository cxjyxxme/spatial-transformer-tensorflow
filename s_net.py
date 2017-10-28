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
from spatial_transformer import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import cv2
from resnet import *
import get_data
from config import *
import time

def get_theta_loss(theta):
    theta = tf.reshape(theta, (-1, 3, 3))
    theta = tf.cast(theta, 'float32')

    d = crop_rate
    target = tf.constant([-d, d, -d, d, -d, -d, d, d], shape=[8], dtype=tf.float32)
    target = tf.tile(target, tf.stack([batch_size]))
    target = tf.reshape(target, tf.stack([batch_size, 2, -1]))

    grid = tf.constant([-1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1], shape=[12], dtype=tf.float32)
    grid = tf.tile(grid, tf.stack([batch_size]))
    grid = tf.reshape(grid, tf.stack([batch_size, 3, -1]))

    T_g = tf.matmul(theta, grid)
    output = tf.slice(T_g, [0, 0, 0], [-1, 2, -1])
    return tf.reduce_mean(tf.abs(output - target))

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

        with tf.variable_scope('resnet', reuse=reuse): 
            config = {'stage_sizes' : [3, 4, 23], 'channel_params' : [  {'kernel_sizes':[1, 3, 1], 'channel_sizes':[64, 64, 256]}, 
                                                                        {'kernel_sizes':[1, 3, 1], 'channel_sizes':[128, 128, 512]}, 
                                                                        {'kernel_sizes':[1, 3, 1], 'channel_sizes':[256, 256, 1024]}]}
            resnet = inference(x_tensor, tot_ch, config)

        with tf.variable_scope('fc', reuse=reuse):
            in_channel = resnet.get_shape().as_list()[-1]
            bn_layer = batch_normalization_layer(resnet, in_channel)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            theta = output_layer(global_pool, 8)
            theta = tf.concat([theta, tf.ones([x_batch_size, 1], tf.float32)], 1)

        with tf.name_scope('theta_loss'):
            use_theta_loss = tf.placeholder(tf.float32)
            theta_loss = get_theta_loss(theta) * use_theta_loss
        regu_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regu_loss = tf.add_n(regu_loss)
        out_size = (height, width)
        h_trans = transformer(x, theta, out_size)
        tf.add_to_collection('output', h_trans)
        tf.summary.image('result', h_trans)
        img_loss = tf.nn.l2_loss(h_trans - y) / batch_size
        total_loss = theta_loss * theta_mul + img_loss * img_mul + regu_loss * regu_mul
        '''
        with tf.name_scope('loss'):
            tf.summary.scalar('tot_loss',total_loss)
            tf.summary.scalar('theta_loss',theta_loss * theta_mul)
            tf.summary.scalar('img_loss',img_loss * img_mul)
            tf.summary.scalar('regu_loss',regu_loss * regu_mul)
        '''
    ret = {}
    ret['error'] = tf.abs(h_trans - y)
    ret['theta_loss'] = theta_loss * theta_mul
    ret['img_loss'] = img_loss * img_mul
    ret['regu_loss'] = regu_loss * regu_mul
    ret['x_tensor'] = x_tensor
    ret['y'] = y
    ret['output'] = h_trans
    ret['total_loss'] = total_loss
    ret['use_theta_loss'] = use_theta_loss
    return ret
