import os
import shutil

os.mkdir('icpr2/image_100')
os.mkdir('icpr2/txt_100')
images = os.listdir('icpr2/image_1000')
count = 0
for image_file in images:
	shutil.copy('icpr2/image_1000/' + image_file,
				'icpr2/image_100/' + image_file)
	shutil.copy('icpr2/txt_1000/'  + image_file[:-4] + '.txt',
				'icpr2/txt_100/'  + image_file[:-4] + '.txt')
	count += 1
	if count == 100:
		break