import tensorflow as tf
import numpy as np
from config import *
from PIL import Image
import cv2
import time

start_with_stable = False#True

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model_name = 'model-45000'
new_saver = tf.train.import_meta_graph(model_dir + model_name + '.meta')
new_saver.restore(sess, model_dir + model_name)
graph = tf.get_default_graph()
x_tensor = graph.get_tensor_by_name('stable_net/input/x_tensor:0')
#output = graph.get_tensor_by_name('stable_net/SpatialTransformer/_transform/Reshape_7:0')
#black_pix = graph.get_tensor_by_name('stable_net/SpatialTransformer/_transform/Reshape_6:0')
output = graph.get_tensor_by_name('stable_net/inference/SpatialTransformer/_transform/Reshape_3:0')
black_pix = graph.get_tensor_by_name('stable_net/inference/SpatialTransformer/_transform/Reshape_2:0')
#black_pix = graph.get_tensor_by_name('stable_net/img_loss/StopGradient:0')

#list_f = open('data_video/test_list_deploy', 'r')
list_f = open('data_video/test_list', 'r')
temp = list_f.read()
video_list = temp.split('\n')

list_f = open('data_video/train_list_deploy', 'r')
temp = list_f.read()
video_list.extend(temp.split('\n'))

for video_name in video_list:
    if (video_name == ""):
        continue
    print(video_name)
    unstable_cap = cv2.VideoCapture('data_video/unstable/' + video_name)  
    fps = unstable_cap.get(cv2.CAP_PROP_FPS)
    print('data_video/unstable/' + video_name)
    videoWriter = cv2.VideoWriter('data_video/output/' + video_name, 
            cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))  
    before_frames = []
    after_frames = []
    print(video_name)
    if (not start_with_stable):
        ret, frame = unstable_cap.read()
        for i in range(before_ch):
            before_frames.append(cvt_img2train(frame, crop_rate))
        for i in range(after_ch + 1):
            ret, frame = unstable_cap.read()
            after_frames.append(cvt_img2train(frame, 1))

        temp = before_frames[0]
        temp = ((np.reshape(temp, (height, width)) + 0.5) * 255).astype(np.uint8)
        videoWriter.write(cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR))
    else:
        stable_cap = cv2.VideoCapture('data_video/stable/' + video_name) 
        for i in range(before_ch):
            ret, frame = unstable_cap.read()
            ret, frame = stable_cap.read()
            before_frames.append(cvt_img2train(frame, crop_rate))
             
            temp = before_frames[i]
            temp = ((np.reshape(temp, (height, width)) + 0.5) * 255).astype(np.uint8)
            videoWriter.write(cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR))
        for i in range(after_ch + 1):
            ret, frame = unstable_cap.read()
            after_frames.append(cvt_img2train(frame, 1))
    len = 0
    while(True):
        in_x = before_frames[0]
        for i in range(1, before_ch):
            in_x = np.concatenate((in_x, before_frames[i]), axis = 3)
        for i in range(after_ch + 1):
            in_x = np.concatenate((in_x, after_frames[i]), axis = 3)
        '''
        in_x_t = in_x
        for i in range(batch_size - 1):
            in_x_t = np.concatenate((in_x_t, in_x), axis = 0)
        '''
        img, black = sess.run([output, black_pix], feed_dict={x_tensor:in_x})
        black = black[0, :, :]
        img = img[0, :, :, :].reshape(height, width)
        frame = img + black * (-1)
        frame = frame.reshape(1, height, width, 1)
        img = ((np.reshape(img, (height, width)) + 0.5) * 255).astype(np.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        videoWriter.write(img)

        ret, frame_unstable = unstable_cap.read() 
        if (not ret):
            break
        len = len + 1
        if (len % 10 == 0):
            print("len: " + str(len))       
        before_frames.append(frame)
        before_frames.pop(0)
        after_frames.append(cvt_img2train(frame_unstable, 1))
        after_frames.pop(0)

        #if (len == 100):
        #    break
    videoWriter.release()
    unstable_cap.release()
