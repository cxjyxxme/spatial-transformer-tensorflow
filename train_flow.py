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
from spatial_transformer import *
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import cv2
from resnet import *
import get_data_flow
from config import *
import time
import s_net

ret1 = s_net.inference_stable_net(False)
ret2 = s_net.inference_stable_net(True)

with tf.name_scope('data_flow'):
    flow = tf.placeholder(tf.float32, [None, height, width, 2])
    x_flow = tf.slice(flow, [0, 0, 0, 0], [-1, -1, -1, 1])
    y_flow = tf.slice(flow, [0, 0, 0, 1], [-1, -1, -1, 1])

with tf.name_scope('temp_loss'):
    use_temp_loss = tf.placeholder(tf.float32)
    output2_aft_flow = interpolate(ret2['output'], x_flow, y_flow, (height, width))
    temp_loss = tf.nn.l2_loss(ret1['output'] - output2_aft_flow) / batch_size * use_temp_loss
with tf.name_scope('errors'):
    tf.summary.image('error_temp', tf.abs(ret1['output'] - output2_aft_flow))
    tf.summary.image('error_1', ret1['error'])
    tf.summary.image('error_2', ret2['error'])

with tf.name_scope('test_flow'):
    warped_y2 = interpolate(ret2['y'], x_flow, y_flow, (height, width))
    tf.summary.image('error_black_wy2', tf.abs(ret1['y'] - warped_y2))
    tf.summary.image('error_black_nowarp', tf.abs(ret2['y'] - ret1['y']))

loss_displayer = tf.placeholder(tf.float32)
with tf.name_scope('test_loss'):
    tf.summary.scalar('test_loss', loss_displayer, collections=['test'])

total_loss = ret1['total_loss'] + ret2['total_loss'] + temp_loss * temp_mul
with tf.name_scope('train_loss'):
    tf.summary.scalar('theta_loss', ret1['theta_loss'] + ret2['theta_loss'])
    tf.summary.scalar('img_loss', ret1['img_loss'] + ret2['img_loss'])
    tf.summary.scalar('regu_loss', ret1['regu_loss'] + ret2['regu_loss'])
    tf.summary.scalar('temp_loss', temp_loss * temp_mul)
    tf.summary.scalar('total_loss', total_loss)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=step_size,decay_rate=0.1, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate)
optimizer = opt.minimize(total_loss, global_step=global_step)


with tf.name_scope('datas'):
    data_x1, data_y1, data_x2, data_y2, data_flow = get_data_flow.read_and_decode(
            "data/train.tfrecords", int(training_iter * batch_size / train_data_size) + 2)
    test_x1, test_y1, test_x2, test_y2, test_flow = get_data_flow.read_and_decode(
            "data/test.tfrecords", int(training_iter * batch_size * test_batches / test_data_size / test_freq) + 2)

    x1_batch, y1_batch, x2_batch, y2_batch, flow_batch = tf.train.shuffle_batch(
                                                [data_x1, data_y1, data_x2, data_y2, data_flow],
                                                batch_size=batch_size, capacity=600,
                                                min_after_dequeue=500, num_threads=2)
    test_x1_batch, test_y1_batch, test_x2_batch, test_y2_batch, test_flow_batch = tf.train.shuffle_batch(
                                                [test_x1, test_y1, test_x2, test_y2, test_flow],
                                                batch_size=batch_size, capacity=600,
                                                min_after_dequeue=500)

merged = tf.summary.merge_all()
test_merged = tf.summary.merge_all("test")
saver = tf.train.Saver()

init_all = tf.initialize_all_variables()

sv = tf.train.Supervisor(logdir='log', save_summaries_secs=0, saver=None)
with sv.managed_session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95))) as sess:
    sess.run(init_all)
    #sess.run(tf.initialize_local_variables())
    threads = tf.train.start_queue_runners(sess=sess)

    time_start = time.time()
    tot_time = 0

    for i in range(training_iter):
        batch_x1s, batch_y1s, batch_x2s, batch_y2s, batch_flows = sess.run(
            [x1_batch, y1_batch, x2_batch, y2_batch, flow_batch])
        if (i > no_theta_iter):
            use_theta = 0
        else:
            use_theta = 1
        if (i >= do_temp_loss_iter):
            use_temp = 1
        else:
            use_temp = 0
        if i % disp_freq == 0:
            print('time:' + str(tot_time) + 's')
            tot_time = 0
            time_start = time.time()
            loss, summary = sess.run([total_loss, merged],
                            feed_dict={
                                ret1['x_tensor']: batch_x1s,
                                ret1['y']: batch_y1s,
                                ret2['x_tensor']: batch_x2s,
                                ret2['y']: batch_y2s,
                                flow: batch_flows,
                                ret1['use_theta_loss']: use_theta,
                                ret2['use_theta_loss']: use_theta,
                                use_temp_loss: use_temp
                            })
            sv.summary_writer.add_summary(summary, i)
            print('Iteration: ' + str(i) + ' Loss: ' + str(loss))
            lr = sess.run(learning_rate)
            print(lr)
            time_end = time.time()
            print('disp time:' + str(time_end - time_start) + 's')
        if i % test_freq == 0:
            sum_test_loss = 0.0
            for j in range(test_batches):
                test_batch_x1s, test_batch_y1s, test_batch_x2s, test_batch_y2s, test_batch_flows = sess.run(
                    [test_x1_batch, test_y1_batch, test_x2_batch, test_y2_batch, test_flow_batch])
                loss = sess.run(total_loss,
                            feed_dict={
                                ret1['x_tensor']: test_batch_x1s,
                                ret1['y']: test_batch_y1s,
                                ret2['x_tensor']: test_batch_x2s,
                                ret2['y']: test_batch_y2s,
                                flow: test_batch_flows,
                                ret1['use_theta_loss']: use_theta,
                                ret2['use_theta_loss']: use_theta,
                                use_temp_loss: use_temp
                            })

                sum_test_loss += loss
            sum_test_loss /= test_batches
            print("Test Loss: " + str(sum_test_loss))
            summary = sess.run(test_merged,
                    feed_dict={
                        loss_displayer: sum_test_loss
                    })
            sv.summary_writer.add_summary(summary, i)
        if i % save_freq == 0:
            saver.save(sess, 'models/model', global_step=i)
        time_end = time.time()
        tot_time += time_end - time_start
        sess.run(optimizer,
                    feed_dict={
                        ret1['x_tensor']: batch_x1s,
                        ret1['y']: batch_y1s,
                        ret2['x_tensor']: batch_x2s,
                        ret2['y']: batch_y2s,
                        flow: batch_flows,
                        ret1['use_theta_loss']: use_theta,
                        ret2['use_theta_loss']: use_theta,
                        use_temp_loss: use_temp
                    })
        time_start = time.time()

