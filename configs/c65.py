import numpy as np
height = 288
width = 512
grid_h = 1
grid_w = 1
batch_size = 10
initial_learning_rate = 2e-4
theta_mul = 400
regu_mul = 30
img_mul = height * width * 1
temp_mul = height * width * 30
black_mul = 10000
distortion_mul = 0#2500000
consistency_mul = 0#50000000
feature_mul = 0#10 * width
id_mul = 10
training_iter = 100000
step_size = 40000
train_data_size = 27000
test_data_size = 2500
crop_rate = 1
before_ch = 5
after_ch = 0
tot_ch = before_ch + after_ch + 1
test_batches = 10
random_crop_rate = 0.9
disp_freq = 200
test_freq = 1000
save_freq = 10000
no_theta_iter = 1000000
do_temp_loss_iter = 5000
do_theta_10_iter = -1
do_black_loss_iter = 1000
do_theta_only_iter = 300
tfrecord_item_num = 10
log_dir = 'log/65/'
model_dir = 'models/65/'
restore_model_dir = 'models/63'
data_dir = '/home/lazycal/workspace/qudou/data8/'
start_step = 55000
rand_H_max = np.array([[1.1, 0.1, 0.5], [0.1, 1.1, 0.5], [0.1, 0.1, 1]])
rand_H_min = np.array([[0.9, -0.1, -0.5], [-0.1, 0.9, -0.5], [-0.1, -0.1, 1]])
max_matches = 3000
