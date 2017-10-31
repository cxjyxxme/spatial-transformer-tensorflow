import tensorflow as tf
import scipy.misc
import random
from config import *

def get_rand_para(seed): 
    h = int(height / random_crop_rate)
    w = int(width / random_crop_rate)
    hh = tf.random_uniform([], minval=0, maxval=h - height, dtype=tf.int32, seed=seed)
    ww = tf.random_uniform([], minval=0, maxval=w - width, dtype=tf.int32, seed=seed)
    return {"h": hh, "w": ww, "flip": (hh + ww) % 2}

def warp_img(image, seed, para):
    h = int(height / random_crop_rate)
    w = int(width / random_crop_rate)
    image = tf.image.resize_images(image, (h, w), method=tf.image.ResizeMethod.BILINEAR)
    image = tf.slice(image, [para['h'], para['w'], 0], [height, width, 1])
    
    image = tf.cond(tf.equal(para['flip'], 0), lambda: image, lambda: tf.image.flip_left_right(image))
    
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5, seed = seed)
    image = tf.image.random_brightness(image, max_delta=32./255., seed = seed)
    ''' #random noise
    noise = np.random.normal(0,0.05,image.shape)
    image = image + noise
    '''

    return tf.clip_by_value(image, -0.5, 0.5)

def warp_flow(flow, para):
    flow_x = tf.slice(flow, [0, 0, 0], [-1, -1, 1])
    flow_y = tf.slice(flow, [0, 0, 1], [-1, -1, 1])
    h = int(height / random_crop_rate)
    w = int(width / random_crop_rate)
    flow_x = tf.image.resize_images(flow_x, (h, w), method=tf.image.ResizeMethod.BILINEAR)
    flow_y = tf.image.resize_images(flow_y, (h, w), method=tf.image.ResizeMethod.BILINEAR)
    flow_x = tf.slice(flow_x, [para['h'], para['w'], 0], [height, width, 1])
    flow_y = tf.slice(flow_y, [para['h'], para['w'], 0], [height, width, 1])
    flow_x = (flow_x + (1 - tf.cast(para['w'], tf.float32) / w * 2)) / (height / float(h)) - 1
    flow_y = (flow_y + (1 - tf.cast(para['h'], tf.float32) / h * 2)) / (width / float(w)) - 1

    fliped_y = tf.image.flip_left_right(flow_y)
    fliped_x = tf.image.flip_left_right(flow_x) * (-1) - 1.0 / width

    flow_x = tf.cond(tf.equal(para['flip'], 0), lambda: flow_x, lambda: fliped_x)
    flow_y = tf.cond(tf.equal(para['flip'], 0), lambda: flow_y, lambda: fliped_y)
    return tf.concat([flow_x, flow_y], axis=2)

def read_and_decode(filepath, num_epochs):
    file_obj = open(filepath + 'list.txt')
    file_txt = file_obj.read()
    file_list = []
    for f in file_txt.split(' '):
        file_list.append(filepath + f)
    filename_queue = tf.train.string_input_producer(file_list, num_epochs=num_epochs, shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'stable': tf.FixedLenFeature([height * width * (before_ch + 2)], tf.float32),
                                           'unstable': tf.FixedLenFeature([height * width * (after_ch + 2)], tf.float32),
                                           'flow': tf.FixedLenFeature([height * width * 2], tf.float32)
                                       })
    stable_ = tf.reshape(features['stable'], [height, width, before_ch + 2])
    unstable_ = tf.reshape(features['unstable'], [height, width, after_ch + 2])
    flow_ = tf.reshape(features['flow'], [height, width, 2])
    seed = random.randint(0, 2**31 - 1)
    para = get_rand_para(seed) 
    for i in range(before_ch + 2):
        temp = tf.slice(stable_, [0, 0, i], [-1, -1, 1])
        if (i == 0):
            stable = warp_img(temp, seed, para)
        else:
            stable = tf.concat([stable, warp_img(temp, seed, para)], 2)
    for i in range(after_ch + 2):
        temp = tf.slice(unstable_, [0, 0, i], [-1, -1, 1])
        if (i == 0):
            unstable = warp_img(temp, seed, para)
        else:
            unstable = tf.concat([unstable, warp_img(temp, seed, para)], 2)
    x1 = tf.concat([tf.slice(stable, [0, 0, 0], [-1, -1, before_ch]), 
                    tf.slice(unstable, [0, 0, 0], [-1, -1, after_ch + 1])], 2)
    y1 = tf.slice(stable, [0, 0, before_ch], [-1, -1, 1])
    x2 = tf.concat([tf.slice(stable, [0, 0, 1], [-1, -1, before_ch]), 
                    tf.slice(unstable, [0, 0, 1], [-1, -1, after_ch + 1])], 2)
    y2 = tf.slice(stable, [0, 0, before_ch + 1], [-1, -1, 1])
    flow = warp_flow(flow_, para)
    return x1, y1, x2, y2, flow

def run():
    x, y = read_and_decode("data/train.tfrecords", 3)

    x_batch, y_batch = tf.train.shuffle_batch([x, y],
                                                    batch_size=30, capacity=2000,
                                                    min_after_dequeue=1000)
    init = tf.initialize_all_variables()
    coord = tf.train.Coordinator()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.initialize_local_variables())
        threads = tf.train.start_queue_runners(sess=sess, coord = coord)
        x_b, y_b = sess.run([x_batch, y_batch])
        print(x_b.shape)
        print(x_b)
        mage_summary = tf.summary.image('y', y_b, 5)
        for i in range(tot_ch):
            temp = tf.slice(x_b, [0, 0, 0, i], [-1, -1, -1, 1])
            mage_summary = tf.summary.image('x' + str(i), temp, 5)
        
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./log/', sess.graph)
        summary_all = sess.run(merged)
        summary_writer.add_summary(summary_all, 0)
        summary_writer.close()
