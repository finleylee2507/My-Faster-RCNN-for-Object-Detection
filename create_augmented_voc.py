import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
import keras_frcnn.vgg as nn #change this to whatever network you're using 
from keras_frcnn import roi_helpers
from keras_frcnn import data_generators
from keras_frcnn import data_augment_testing
from shutil import copyfile


def roi_occlusion(img_data, img):
	
	data=img_data['bboxes']
	
	#iterate over each object 
	for d in data:
	  xmin=d['x1']
	  xmax=d['x2']
	  ymin=d['y1']
	  ymax=d['y2']

		
	  img=data_augment_testing.img_augmented_occlusion_v2(img,xmin,xmax,ymin,ymax,p=1,pixel_level=False)
	
	return img







sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
				default="pascal_voc")
parser.add_option("--data_set",dest="data_set",default="VOC2007")
parser.add_option("--augmentation",dest="augmentation",default="random_occlusion")
(options, args) = parser.parse_args()

(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')


if options.parser == 'pascal_voc':
	from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
	from keras_frcnn.simple_parser import get_data
else:
	raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)


img_path = options.test_path


class_mapping = C.class_mapping

new_img_path=os.path.join(options.test_path,options.data_set,"augmentedImages")

if(os.path.exists(new_img_path)):
	#remove all exisiting files
	print('Removing all exisiting files.')
	for f in os.listdir(new_img_path):
		os.remove(os.path.join(new_img_path,f))
else:
	os.makedirs(new_img_path)


all_imgs, _, _ = get_data(options.test_path,C,mode='test')
test_imgs = [s for s in all_imgs if s['imageset'] == 'test']



T = {}
P = {}


for idx, img_data in enumerate(test_imgs):
	print('{}/{}'.format(idx,len(test_imgs)))
	
	filepath = img_data['filepath']
	img = cv2.imread(filepath)
	if(options.augmentation=='random_occlusion'):	
		img=data_augment_testing.img_augment_occlusion_v1(img)
	if(options.augmentation=='roi_occlusion'):
		img = roi_occlusion(img_data, img)
	if(options.augmentation=='noise'):
		img=data_augment_testing.img_augment_noise(img)

	
	
	
	cv2.imwrite(os.path.join(new_img_path,os.path.basename(filepath)),img)
	


	

	
 