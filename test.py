import tensorflow as tf
import scipy.misc

def cvt2img(img):
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [288, 512, 1])
#    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    return img

def read_and_decode(filename, num_epochs):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'unstable' : tf.FixedLenFeature([], tf.string),
                                           's0' : tf.FixedLenFeature([], tf.string),
                                           's1' : tf.FixedLenFeature([], tf.string),
                                           's2' : tf.FixedLenFeature([], tf.string),
                                           's3' : tf.FixedLenFeature([], tf.string),
                                           's4' : tf.FixedLenFeature([], tf.string),
                                       })
    unstable_frame = cvt2img(features['unstable'])
    s0 = cvt2img(features['s0'])
    s1 = cvt2img(features['s1'])
    s2 = cvt2img(features['s2'])
    s3 = cvt2img(features['s3'])
    s4 = cvt2img(features['s4'])
    
    return unstable_frame, s0, s1, s2, s3, s4

unstable_frame, s0, s1, s2, s3, s4 = read_and_decode("data/train.tfrecords", 3)

unstable_batch, s0_batch, s1_batch, s2_batch, s3_batch, s4_batch = tf.train.shuffle_batch([unstable_frame, s0, s1, s2, s3, s4],
                                                batch_size=30, capacity=2000,
                                                min_after_dequeue=1000)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    u_b, s0_b, s1_b, s2_b, s3_b, s4_b = sess.run([unstable_batch, s0_batch, s1_batch, s2_batch, s3_batch, s4_batch])
    print(u_b.shape)
    mage_summary = tf.summary.image('s0_b', s0_b, 1)
    mage_summary = tf.summary.image('s1_b', s1_b, 1)
    mage_summary = tf.summary.image('s2_b', s2_b, 1)
    mage_summary = tf.summary.image('s3_b', s3_b, 1)
    mage_summary = tf.summary.image('s4_b', s4_b, 1)
    mage_summary = tf.summary.image('u_b', u_b, 1)
    
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./log/', sess.graph)
    summary_all = sess.run(merged)
    summary_writer.add_summary(summary_all, 0)
    summary_writer.close()
