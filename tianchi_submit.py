import os

from tqdm import tqdm
from network import East
from predict import predict_txt
import argparse
parser = argparse.ArgumentParser(description='options')
parser.add_argument('--section', type=str, default='local',
                    help='cfg to load')
args = parser.parse_args()

if args.section == 'local':
    import cfg_local as cfg
if args.section == 'server':
    import cfg_server as cfg

from keras import backend as K
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_to_use
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = cfg.per_process_gpu_memory_fraction
session = tf.Session(config=config)
K.set_session(session)

if __name__ == '__main__':
    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)

    # not getting results for challenge 3
    image_test_dir = os.path.join(cfg.data_dir, 'icpr_mtwi_task2/image_test/')
    txt_test_dir = os.path.join(cfg.data_dir, 'icpr_mtwi_task2/txt_test_%s/'%(cfg.train_task_id))
    if not os.path.exists(txt_test_dir):
        os.mkdir(txt_test_dir)
    test_imgname_list = os.listdir(image_test_dir)
    print('found %d test images.' % len(test_imgname_list))
    for test_img_name, _ in zip(test_imgname_list,
                                tqdm(range(len(test_imgname_list)))):
        img_path = os.path.join(image_test_dir, test_img_name)
        txt_path = os.path.join(txt_test_dir, test_img_name[:-4] + '.txt')
        predict_txt(east_detect, img_path, txt_path, cfg.pixel_threshold, True)
