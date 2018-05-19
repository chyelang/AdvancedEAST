gpu_to_use = '7'
per_process_gpu_memory_fraction = 1
import os
train_task_id = '1RC513'
load_weights = False
train_task_id_to_reload_weights = ''
#backbone = 'vgg16'
backbone = 'resnet50'
initial_epoch = 0
epoch_num = 24
lr = 1e-3
decay = 5e-4
# clipvalue = 0.5  # default 0.5, 0 means no clip
patience = 2
lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

total_img = 10000
validation_split_ratio = 0.1
max_train_img_size = int(train_task_id[-3:])
max_predict_img_size = int(train_task_id[-3:])  # 2400
assert max_train_img_size in [257, 385, 513, 641, 737], \
    'max_train_img_size must in [257, 385, 513, 641, 737]'
# assert max_train_img_size in [256, 384, 512, 640, 736], \
#     'max_train_img_size must in [256, 384, 512, 640, 736]'
if max_train_img_size == 257:
    batch_size = 8
elif max_train_img_size == 385:
    batch_size = 4
elif max_train_img_size == 513:
    batch_size = 2
else:
    batch_size = 8
steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size
validation_steps = total_img * validation_split_ratio // batch_size

data_dir = '/data/chenxinyang/icpr2'
origin_image_dir_name = 'image_10000/'
origin_txt_dir_name = 'txt_10000/'
train_image_dir_name = 'images_%s/' % max_train_img_size
train_label_dir_name = 'labels_%s/' % max_train_img_size
show_gt_image_dir_name = 'show_gt_images_%s/' % max_train_img_size
show_act_image_dir_name = 'show_act_images_%s/' % max_train_img_size
gen_origin_img = True
draw_gt_quad = False
draw_act_quad = False
val_fname = 'val_%s.txt' % max_train_img_size
train_fname = 'train_%s.txt' % max_train_img_size
# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.2
# pixels between 0.1 and 0.3 are side pixels
shrink_side_ratio = 0.6
epsilon = 1e-4

num_channels = 3
feature_layers_range = range(5, 1, -1)
# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)
# pixel_size = 4
pixel_size = 2 ** feature_layers_range[-1]
locked_layers = False

model_weights_path = '/data/chenxinyang/AdvancedEAST/model/weights_%s.{epoch:03d}-{val_loss:.3f}.h5' \
                     % train_task_id
saved_model_file_path = '/data/chenxinyang/AdvancedEAST/saved_model/east_model_%s.h5' % train_task_id
saved_model_weights_file_path = '/data/chenxinyang/AdvancedEAST/saved_model/east_model_weights_%s.h5'\
                                % train_task_id
model_weights_file_path_to_reload = 'data/chenxinyang/AdvancedEAST/saved_model/east_model_weights_%s.h5'\
                                % train_task_id_to_reload_weights

if not os.path.exists(os.path.dirname(model_weights_path)):
    os.mkdir(os.path.dirname(model_weights_path))
if not os.path.exists(os.path.dirname(saved_model_file_path)):
    os.mkdir(os.path.dirname(saved_model_file_path))
if not os.path.exists(os.path.dirname(saved_model_weights_file_path)):
    os.mkdir(os.path.dirname(saved_model_weights_file_path))

pixel_threshold = 0.9
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
predict_cut_text_line = False
predict_write2txt = True
