import tensorflow as tf
import numpy as np
from config import *
from PIL import Image
import cv2
import time

sess = tf.Session()

new_saver = tf.train.import_meta_graph('models/model-49000.meta')
new_saver.restore(sess, 'models/model-49000')
graph = tf.get_default_graph()
x_tensor = graph.get_operation_by_name('input/x_tensor').outputs[0]
output = tf.get_collection('output')[0]


list_f = open('data_video/list', 'rw+')
temp = list_f.read()
video_list = temp.split('\n')

for video_name in video_list:
    if (video_name == ""):
        break
    unstable_cap = cv2.VideoCapture('data_video/unstable/' + video_name)  
    fps = unstable_cap.get(cv2.cv.CV_CAP_PROP_FPS)  
    videoWriter = cv2.VideoWriter('data_video/output/' + video_name, 
            cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, (width, height))  
    train_frames = []
    print(video_name)
    #debug
    stable_cap = cv2.VideoCapture('data_video/stable/' + video_name)  
    for i in range(4):
        ret, frame = unstable_cap.read()
        ret, frame = stable_cap.read()
        train_frames.append(cvt_img2train(frame, crop_rate))
         
        temp = train_frames[i]
        temp = ((np.reshape(temp, (height, width)) + 0.5) * 255).astype(np.uint8)
        videoWriter.write(cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR))
    #debug
    '''
    ret, frame = unstable_cap.read()
    for i in range(4):
        train_frames.append(cvt_img2train(frame, crop_rate))
    
    temp = train_frames[0]
    temp = ((np.reshape(temp, (height, width)) + 0.5) * 255).astype(np.uint8)
    videoWriter.write(cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR))
    '''
    len = 0
    while(True):
        ret, frame_stable = stable_cap.read()#debug
        ret, frame_unstable = unstable_cap.read()
        if (not ret):
            break
        len = len + 1
        if (len % 10 == 0):
            print("len: " + str(len))
        #if (len == 100):
        #    break
        frame = cvt_img2train(frame_unstable, 1)
        in_x_t = np.concatenate((train_frames[0], train_frames[1], train_frames[2], train_frames[3], frame), axis=3)
        in_x = in_x_t
        for i in range(batch_size - 1):
            in_x = np.concatenate((in_x, in_x_t), axis=0)
        img = sess.run(output, feed_dict={x_tensor:in_x})
        img = img[0, :, :, :]
        #train_frames.append(img.reshape(1, height, width, 1))
        train_frames.append(cvt_img2train(frame_stable, crop_rate))#debug
        train_frames.pop(0)
        img = ((np.reshape(img, (height, width)) + 0.5) * 255).astype(np.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        videoWriter.write(img)
    videoWriter.release()
    unstable_cap.release()
