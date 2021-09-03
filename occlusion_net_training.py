from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import os
import pandas as pd
import math 
import keras
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model, load_model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from matplotlib import pyplot as plt
from keras_frcnn import vgg_occlusion_net as occlusion


if 'tensorflow' == K.backend():
	import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config2 = tf.ConfigProto()
config2.gpu_options.allow_growth = True
set_session(tf.Session(config=config2))
sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path",
				  help="Path to training data.")
parser.add_option("--rp", "--record_path", dest="record_path",
				  help="Path to the record path.", default='./models/records/record_vgg_voc_occlusion_net.csv')
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
				  default="pascal_voc")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				  help="Number of RoIs to process at once.", default=10)
parser.add_option("--network", dest="network",
				  help="Base network to use. Supports vgg or resnet50.", default='vgg')
parser.add_option("--hf", dest="horizontal_flips",
				  help="Augment with horizontal flips in training. (Default=true).", action="store_false", default=True)
parser.add_option("--vf", dest="vertical_flips",
				  help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
				  action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs",
				  help="Number of epochs.", default=50)
parser.add_option("--config_filename", dest="config_filename", help="Location to store all the metadata related to the training (to be used when testing).",
				  default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path",
				  help="Output path for weights.", default='./models/vgg/vgg_occlusion_net.hdf5')
parser.add_option("--input_weight_path_rcnn", dest="input_weight_path_rcnn",
				  help="Input path for weights (rcnn). If not specified, will try to load default weights provided by keras.", default=None)
parser.add_option("--rpn", dest="rpn_weight_path",
				  help="Input path for rpn.", default=None)
parser.add_option("--opt", dest="optimizers",
				  help="set the optimizer to use", default="SGD")
parser.add_option("--elen", dest="epoch_length",
				  help="set the epoch length. def=1000", default=1000)
parser.add_option("--load_rcnn", dest="load_rcnn",
				  help="What model to load for the rcnn", default=None)
parser.add_option("--dataset", dest="dataset",
				  help="name of the dataset", default="voc")
parser.add_option("--cat", dest="cat",
				  help="categroy to train on. default train on all cats.", default=None)
parser.add_option("--lr", dest="lr", help="learn rate",
				  type=float, default=1e-3)
parser.add_option("--input_weight_path_occlusion", dest="input_weight_path_occlusion",
				  help="Input path for weights (occlusion net). If not specified, will try to load default weights provided by keras.", default=None)
parser.add_option("--load_occlusion", dest="load_occlusion",
				  help="What model to load for the occlusion net", default=None)
parser.add_option("--occlusion_conv_output_channel", dest="occlusion_conv_output_channel",
				  help="The output channel size for the conv layer in the occlusion network", type=int, default=512)

(options, args) = parser.parse_args()

if not options.train_path:   # if filename is not given
	parser.error(
		'Error: path to training data must be specified. Pass --path to command line')
if options.parser == 'pascal_voc':
	from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
	from keras_frcnn.simple_parser import get_data
else:
	raise ValueError(
		"Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object


config_output_filename = options.config_filename

# if we already have a config file, load it
if os.path.exists(config_output_filename):
	print('Previous config file loaded.')
	with open(config_output_filename, 'rb') as f_in:
		C = pickle.load(f_in)
else:
	C = config.Config()
	C.class_mapping = {}


C.use_horizontal_flips = bool(options.horizontal_flips)
print("Horizontal flips?")
print(C.use_horizontal_flips)

C.use_vertical_flips = bool(options.vertical_flips)
print("Vertical flips? ")
print(C.use_vertical_flips)
C.rot_90 = bool(options.rot_90)
print("Rotational? ")
print(C.rot_90)

# mkdir to save models. (for example if vgg is chosen, a folder name "vgg" will be created)
if not os.path.isdir("models"):
	os.mkdir("models")
if not os.path.isdir("models/"+options.network):
	os.mkdir(os.path.join("models", options.network))

# C.model_path = os.path.join(
# 	"models", options.network, options.dataset+"_occlusion_net"+".hdf5")
C.model_path=options.output_weight_path
C.record_path = options.record_path  # get the path to store the record
C.num_rois = int(options.num_rois)

# we will use resnet. may change to others
if options.network == 'vgg' or options.network == 'vgg16':
	C.network = 'vgg16'
	from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
	from keras_frcnn import resnet as nn
	C.network = 'resnet50'
elif options.network == 'vgg19':
	from keras_frcnn import vgg19 as nn
	C.network = 'vgg19'
elif options.network == 'mobilenetv1':
	from keras_frcnn import mobilenetv1 as nn
	C.network = 'mobilenetv1'
elif options.network == 'mobilenetv2':
	from keras_frcnn import mobilenetv2 as nn
	C.network = 'mobilenetv2'
elif options.network == 'densenet':
	from keras_frcnn import densenet as nn
	C.network = 'densenet'
else:
	print('Not a valid model')
	raise ValueError

print("Input weight path (rcnn): ", options.input_weight_path_rcnn)
print("Loaded model (rcnn): ", options.load_rcnn)

print("Input weight path (occlusion net): ",
	  options.input_weight_path_occlusion)
print("Loaded model (occlusion net): ", options.load_occlusion)

# check if weight path was passed via command line
if options.input_weight_path_rcnn:
	print("Input weight set!")
	C.base_net_weights = options.input_weight_path_rcnn
else:
	# set the path to weights based on backend and model
	print("Using the pretrained weight from github!")
	C.base_net_weights = nn.get_weight_path()


all_imgs, classes_count, class_mapping = get_data(
	options.train_path, C, options.cat, mode='train')

if not os.path.exists(config_output_filename):  # only for new config file
	if 'bg' not in classes_count:
		classes_count['bg'] = 0
		class_mapping['bg'] = len(class_mapping)
		C.class_mapping = class_mapping
else:
	if 'bg' not in classes_count:
		classes_count['bg'] = 0
		C.class_mapping = class_mapping


inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

print('Class mapping: ')
pprint.pprint(inv_map)
config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C, config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
		config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))


data_gen_train = data_generators.get_anchor_gt(
	train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(
	val_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='val')

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
else:
	input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))
new_classifier_part2_input = Input(shape=(1, 7, 7, 512))  # use 1 as num_rois because we need to pass in 1 roi at a time 
# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

# define the rest of the Faster-RCNN

new_classifier_part1 = nn.new_classifier_part1(
	shared_layers, roi_input, C.num_rois)

new_classifier_part2 = nn.new_classifier_part2(
	new_classifier_part2_input, nb_classes=len(classes_count))


model_rpn = Model(img_input, rpn[:2])
# model_classifier = Model([img_input, roi_input], classifier)

# the new stuffs
model_pooling = Model([img_input, roi_input], new_classifier_part1)
model_new_classifier = Model(new_classifier_part2_input, new_classifier_part2)

# the occlusion net model
model_occlusion_net = occlusion.OcclusionNet(
	input_height=7, input_width=7, input_channel=512,conv_output_channel=options.occlusion_conv_output_channel)

# DON'T NEED THIS FOR THE OCCLUSION NET TRAINING
# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
# model_all = Model([img_input, roi_input], rpn[:2] + classifier)
# model_all_new=Model([img_input,new_classifier_part2_input],rpn[:2] + new_classifier_part2)


# load pretrained weights (maybe be overridden later)
try:
	print('loading weights from {}'.format(C.base_net_weights))
	model_rpn.load_weights(C.base_net_weights, by_name=True)
	# model_classifier.load_weights(C.base_net_weights, by_name=True)
	# might not be necessary
	model_pooling.load_weights(C.base_net_weights, by_name=True)
	model_new_classifier.load_weights(C.base_net_weights, by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')


# optimizer setup
if options.optimizers == "SGD":  # default setting (DOESN'T WORK WELL FOR THE OCCLUSION NET)
	optimizer_occlusion = SGD(lr=options.lr/5, decay=0.0005, momentum=0.9)
else:
	optimizer_occlusion = Adam(lr=options.lr, clipnorm=0.001)




if options.load_occlusion is not None:  # Resume training, loading pretrained occlusion network
	print("loading previous occlusion model from ", options.load_occlusion)
	model_occlusion_net.load_weights(options.load_occlusion, by_name=True)

	# load the records
	record_df = pd.read_csv(C.record_path)
	r_curr_loss = record_df['training loss']
	r_accuracy = record_df['accuracy']
	r_elapsed_time = record_df['elapsed time']
	r_precision=record_df['precision']
	r_recall=record_df['recall']
	r_f1=record_df['f1 score']
	print('Already train %dK batches' % (len(record_df)))
	print("Resume training")

elif options.input_weight_path_occlusion is not None: #Initialize training with pretrained weight 
	print("loading pretrained occlusion net weight from, ",
		  options.input_weight_path_occlusion)
	model_occlusion_net.load_weights(
		options.input_weight_path_occlusion, by_name=True)

	# create record table
	record_df = pd.DataFrame(columns=['training loss', 'precision','recall', 'f1 score', 'elapsed time','accuracy'])
else:
	print("No previous occlusion model or pretrained weight loaded")

	# create record table
	record_df = pd.DataFrame(columns=['training loss', 'precision','recall', 'f1 score', 'elapsed time','accuracy'])


# compile the model AFTER loading weights!
model_rpn.compile(optimizer='adam', loss=[losses.rpn_loss_cls(
	num_anchors), losses.rpn_loss_regr(num_anchors)])
# model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(
# 	len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})

# the new network
# compile with random parameters
model_pooling.compile(optimizer='adam', loss='mae')
model_new_classifier.compile(optimizer='adam', loss=[losses.class_loss_cls, losses.class_loss_regr(
	len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})

# the occlusion network
model_occlusion_net.compile(
	optimizer=optimizer_occlusion, loss=losses.weighted_bincrossentropy, metrics=['accuracy',losses.get_precision,losses.get_recall,losses.get_f1])

# model_all.compile(optimizer='sgd', loss='mae')

# the new network
# model_all_new.compile(optimizer='sgd', loss='mae')


model_pooling.summary()
model_new_classifier.summary()
model_occlusion_net.summary()
# training settings


epoch_length = int(options.epoch_length)
num_epochs = int(options.num_epochs)
iter_num = 0

# if(len(record_df) > 0):  # resume training from previous epoch
#     num_epochs -= len(record_df)

losses = np.zeros((epoch_length, 5))
start_time = time.time()

if len(record_df) == 0:
	best_loss = np.Inf
else:
	best_loss = np.min(r_curr_loss)

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True

starting_epoch = 0
if(len(record_df) > 0):  # resume training from the previous epoch
	starting_epoch = len(record_df)

for epoch_num in range(starting_epoch, num_epochs):
	progbar = generic_utils.Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

	# first 3 epoch is warmup
	if epoch_num < 3 and options.rpn_weight_path is not None:
		# K.set_value(model_rpn.optimizer.lr, options.lr/30)
		# K.set_value(model_new_classifier.optimizer.lr, options.lr/3)
		K.set_value(model_occlusion_net.optimizer.lr, options.lr/3)

	# print out the learning rate each epoch for debugging purposes
	print("Occlusion net learning rate: ", K.eval(
		model_occlusion_net.optimizer.lr))

	while True:
		try:

			X, Y, img_data = next(data_gen_train)

			# loss_rpn = model_rpn.train_on_batch(X, Y)

			P_rpn = model_rpn.predict_on_batch(X)
			R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(  # could try out different overlap_threshold
			), use_regr=True, overlap_thresh=0.8, max_boxes=300)
			# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
			X2, Y1, Y2, IouS = roi_helpers.calc_iou(
				R, img_data, C, class_mapping)

			neg_samples = np.where(Y1[0, :, -1] == 1)
			pos_samples = np.where(Y1[0, :, -1] == 0)

			if len(neg_samples) > 0:
				neg_samples = neg_samples[0]
			else:
				neg_samples = []

			if len(pos_samples) > 0:
				pos_samples = pos_samples[0]
			else:
				pos_samples = []

# ###FOR OCCLUSION NET TRAINING, WE ONLY CONSIDER THE POSITIVE ROIS

			if C.num_rois > 1:
				# If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
				if len(pos_samples) < C.num_rois//2:
					selected_pos_samples = pos_samples.tolist()
				else:
					selected_pos_samples = np.random.choice(
						pos_samples, C.num_rois//2, replace=False).tolist()
					# Randomly choose (num_rois - num_pos) neg samples
				try:
					selected_neg_samples = np.random.choice(
						neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
				except:
					selected_neg_samples = np.random.choice(
						neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
					# Save all the pos and neg samples in sel_samples
				# print("Pos: ",selected_pos_samples)
				# print("Neg", selected_neg_samples)
				sel_samples = selected_pos_samples + selected_neg_samples
				# print("All: ",sel_samples)
				#sel_samples = selected_pos_samples
			else:
				# in the extreme case where num_rois = 1, we pick a random pos or neg sample
				selected_pos_samples = pos_samples.tolist()
				selected_neg_samples = neg_samples.tolist()
				if np.random.randint(0, 2):
					sel_samples = random.choice(neg_samples)
				else:
					sel_samples = random.choice(pos_samples)

			pooling_output = model_pooling.predict_on_batch(
				[X, X2[:, sel_samples, :]])
			# print("Pooling output shape", pooling_output.shape)

			# initialize training sample array
			x_occlude = np.empty((len(selected_pos_samples), 7, 7, 512))
			# initialize ground truth array
			y_occlude = np.empty((len(selected_pos_samples), 7, 7, 1))

			# iterate over the n rois

			for i in range(0, C.num_rois):
				sel_sample = sel_samples[i]
				# print(i)
				if(sel_sample in selected_neg_samples): #we skip the negative samples 
					# print("Negative sample found!")
					# print(sel_sample)
					continue

				test1 = pooling_output[:, i]
  

	

				test2 = Y1[:, sel_sample, :]
				test3 = Y2[:, sel_sample, :]

				curr_max_loss = 0
				# ground_truth_mask = np.empty((1, 7, 7, 2))  # initialize ground truth mask
				best_row = 0
				best_col = 0
				x_occlude[i] = test1  # append training sample


			
				
				# iterate over the x and y dimension of the roi and applying a 2 x 2 sliding window on it
				for j in range(0, 6):
					for k in range(0, 6):
						#print("j: ", j, " k: ", k)
						new_test1 = np.copy(test1)
						new_test2 = np.copy(test2)
						new_test3 = np.copy(test3)
						# dropping out values of all the channels in the corresponding spatial location
						new_test1[0, j:j+2, k:k+2, :] = 0
						# print(new_test1[0,:,:,0])
						new_test1 = np.expand_dims(new_test1, axis=0)
						new_test2 = np.expand_dims(new_test2, axis=0)
						new_test3 = np.expand_dims(new_test3, axis=0)

						test_loss = model_new_classifier.test_on_batch(
							new_test1, [new_test2, new_test3])
						test_classfication_loss = test_loss[1]
						# print("Test loss, ROI :", i, "j: ", j,
						# 	  "k: ", k, " loss: ", test_loss)

						if(test_classfication_loss > curr_max_loss):
							curr_max_loss = test_classfication_loss
							best_row = j
							best_col = k

				# print("Best row and column: ",best_row, best_col)
				ground_truth_mask = np.zeros((1, 7, 7, 1))

				for row in range(0, 7):
					for col in range(0, 7):
						# for the window location
						if((row >= best_row and row <= best_row+1) and (col >= best_col and col <= best_col+1)):
							ground_truth_mask[0, row, col, 0] = 1

						else:  # for the other pixels
							ground_truth_mask[0, row, col, 0] = 0
				# print(ground_truth_mask)
				y_occlude[i] = ground_truth_mask

			#check for empty sample 
			if(len(x_occlude)==0):
				continue 


			occlusion_loss = model_occlusion_net.train_on_batch(
				x_occlude, y_occlude)

			# print("Occlusion loss: ", occlusion_loss)
			
			# store the training loss
			losses[iter_num, 0] = occlusion_loss[0]  
			# store the training accuracy
			losses[iter_num, 1] = occlusion_loss[1]
			#store the precision
			losses[iter_num,2]=occlusion_loss[2]
			#store the recall
			losses[iter_num,3]=occlusion_loss[3]
			#store the f1 score
			losses[iter_num,4]=occlusion_loss[4]

			# if(math.isnan(occlusion_loss[0])): #for debugging purpose 
			# 	sys.exit()

			iter_num += 1
		
			progbar.update(iter_num, [('loss', np.nanmean(losses[:iter_num, 0])), ('accuracy', np.nanmean(
				losses[:iter_num, 1])),('precision',np.nanmean(losses[:iter_num,2])),('recall',np.nanmean(losses[:iter_num,3])),('f1 score',np.nanmean(losses[:iter_num,4]))])  # display progress bar

			# at the end of every epoch
			if iter_num == epoch_length:
				epoch_loss = np.nanmean(losses[:, 0])
				epoch_accuracy = np.nanmean(losses[:, 1])
				epoch_precision=np.nanmean(losses[:,2])
				epoch_recall=np.nanmean(losses[:,3])
				epoch_f1=np.nanmean(losses[:,4])

				if C.verbose:
					print('Training loss: {}'.format(epoch_loss))
					print('Training accuracy: {}'.format(epoch_accuracy))
					print('Training precision: {}'.format(epoch_precision))					
					print('Training recall: {}'.format(epoch_recall))
					print('Training f1 score: {}'.format(epoch_f1))
					

					print('Elapsed time: {}'.format(time.time() - start_time))
					elapsed_time = (time.time()-start_time)/60

				curr_loss = epoch_loss
				iter_num = 0
				start_time = time.time()

				if curr_loss < best_loss:
					if C.verbose:
						print('Total loss decreased from {} to {}, saving weights'.format(
							best_loss, curr_loss))
					best_loss = curr_loss

					model_occlusion_net.save_weights(C.model_path)

				new_row = {'training loss': round(
					curr_loss, 3), 'precision':round(epoch_precision,3),'recall':round(epoch_recall,3), 'f1 score':round(epoch_f1,3), 'elapsed time': round(elapsed_time, 3),'accuracy': round(epoch_accuracy, 3)}
				record_df = record_df.append(
					new_row, ignore_index=True)

				record_df.to_csv(C.record_path, index=0)

				break
		except Exception as e:
			print('Exception: {}'.format(e))
			continue

print('Training complete, exiting.')
