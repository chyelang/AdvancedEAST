import numpy as np
from PIL import Image, ImageDraw
import os

txt_dir = '5_23_txt_test_2-2s01856RC513'
result_dir = "result_"+txt_dir.strip().split('_')[-1]
if not os.path.exists('submit/' + result_dir):
    os.mkdir('submit/' + result_dir)

def test():
    image_names = []
    image_files = os.listdir('submit/test_image_samples')
    for image_file in image_files:
        image_names.append(image_file[:-4])
    for image_name in image_names:
        path = 'submit/test_image_samples/%s.jpg'%(image_name)
        format = 'jpeg'
        if not os.path.exists(path):
            path = path[:-4] + ".png"
            format = 'png'
        with Image.open(path) as im:
            # draw on the origin img
            draw = ImageDraw.Draw(im)
            try:
                with open('submit/%s/%s.txt'%(txt_dir, image_name), 'r') as f:
                    anno_list = f.readlines()
            except Exception:
                continue
            for anno in anno_list:
                anno_colums = anno.strip().split(',')
                anno_array = np.array(anno_colums)
                xy_list = np.reshape(anno_array.astype(float), (4, 2))
                draw.line([tuple(xy_list[0]), tuple(xy_list[1]), tuple(xy_list[2]),
                           tuple(xy_list[3]), tuple(xy_list[0])],
                          width=8,
                          fill='green')
            if format == 'jpeg':
                im.save('submit/%s/%s_anno.jpg'%(result_dir, image_name))
            else:
                im.save('submit/%s/%s_anno.png' % (result_dir, image_name))

test()
