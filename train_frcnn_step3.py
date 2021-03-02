#this is step 3 of the alternating training, where we would train the rpn head 

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
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model, load_model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from matplotlib import pyplot as plt 

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
                  help="Path to the record path.", default='./models/records/record_step3.csv')
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                  default="pascal_voc")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                  help="Number of RoIs to process at once.", default=10)
parser.add_option("--network", dest="network",
                  help="Base network to use. Supports vgg or resnet50.", default='vgg')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=true).", action="store_false", default=True)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
				  action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs",
                  help="Number of epochs.", default=50)
parser.add_option("--config_filename", dest="config_filename", help="Location to store all the metadata related to the training (to be used when testing).",
                  default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path",
                  help="Output path for weights.", default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path",
                  help="Input path for weights. If not specified, will try to load default weights provided by keras.", default=None)
parser.add_option("--rpn", dest="rpn_weight_path",
                  help="Input path for rpn.", default=None)
parser.add_option("--opt", dest="optimizers",
                  help="set the optimizer to use", default="Adam")
parser.add_option("--elen", dest="epoch_length",
                  help="set the epoch length. def=1000", default=1000)
parser.add_option("--load", dest="load",
                  help="What model to load", default=None)
parser.add_option("--dataset", dest="dataset",
                  help="name of the dataset", default="voc")
parser.add_option("--cat", dest="cat",
                  help="categroy to train on. default train on all cats.", default=None)
parser.add_option("--lr", dest="lr", help="learn rate",
                  type=float, default=1e-5)

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
C = config.Config()

C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

# mkdir to save models. (for example if vgg is chosen, a folder name "vgg" will be created)
if not os.path.isdir("models"):
    os.mkdir("models")
if not os.path.isdir("models/"+options.network):
    os.mkdir(os.path.join("models", options.network))
C.model_path = os.path.join("models", options.network, options.dataset+"_step3_"+".hdf5")
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

# check if weight path was passed via command line
if options.input_weight_path:
    C.base_net_weights = options.input_weight_path
else:
    # set the path to weights based on backend and model
    C.base_net_weights = nn.get_weight_path()

all_imgs, classes_count, class_mapping = get_data(
    options.train_path, options.cat)

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

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

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=False)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(
    classes_count), trainable=False)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

# load pretrained weights (maybe be overridden later)
try:
    print('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
        https://github.com/fchollet/keras/tree/master/keras/applications')

# optimizer setup
print(options.optimizers)
if options.optimizers == "SGD": #default setting 
    if options.rpn_weight_path is not None: #if using pretrained rpn model 
        optimizer = SGD(lr=options.lr/100, decay=0.0005, momentum=0.9)
        optimizer_classifier = SGD(lr=options.lr/5, decay=0.0005, momentum=0.9)
    else: #if not using pretrained rpn 
        optimizer = SGD(lr=options.lr/10, decay=0.0005, momentum=0.9)
        optimizer_classifier = SGD(
            lr=options.lr/10, decay=0.0005, momentum=0.9)
else:
    optimizer = Adam(lr=options.lr, clipnorm=0.001)
    optimizer_classifier = Adam(lr=options.lr, clipnorm=0.001)
    

# may use this to resume from rpn models or previous training. specify either rpn or frcnn model to load
if options.load is not None:  # with pretrained FRCNN model
    print("loading previous model from ", options.load)
    model_rpn.load_weights(options.load, by_name=True)
    model_classifier.load_weights(options.load, by_name=True)
    # load the records
    record_df = pd.read_csv(C.record_path)
    
    r_loss_rpn_cls = record_df['loss_rpn_cls']
    r_loss_rpn_regr = record_df['loss_rpn_regr']
    r_curr_loss = record_df['curr_loss']
    r_elapsed_time = record_df['elapsed_time']
    r_mAP = record_df['mAP']
    print('Already train %dK batches' % (len(record_df)))

    # Create the record.csv file to record losses, acc and mAP (for first time training on an existing model, delete if resume)
    # record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls',
    #                                   'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
elif options.rpn_weight_path is not None:  # with pretrained RPN
    print("loading RPN weights from ", options.rpn_weight_path)
    model_rpn.load_weights(options.rpn_weight_path, by_name=True)
    record_df = pd.DataFrame(columns=['loss_rpn_cls', 'loss_rpn_regr', 'curr_loss', 'elapsed_time', 'mAP'])
else:
    print("no previous model was loaded")
    # Create the record.csv file to record losses, acc and mAP
    record_df = pd.DataFrame(columns=['loss_rpn_cls', 'loss_rpn_regr', 'curr_loss', 'elapsed_time', 'mAP'])


# compile the model AFTER loading weights!
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(
    num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(
    len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')


#print model summary 

model_rpn.summary()
model_classifier.summary()
model_all.summary()
# training settings


epoch_length = int(options.epoch_length)
num_epochs = int(options.num_epochs)
iter_num = 0

# if(len(record_df) > 0):  # resume training from previous epoch
#     num_epochs -= len(record_df)

losses = np.zeros((epoch_length, 2))
# rpn_accuracy_rpn_monitor = []
# rpn_accuracy_for_epoch = []
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
        K.set_value(model_rpn.optimizer.lr, options.lr/30)
        K.set_value(model_classifier.optimizer.lr, options.lr/3)

    while True:
        try:
           
            X, Y, img_data = next(data_gen_train)
            #train rpn 
            loss_rpn = model_rpn.train_on_batch(X, Y)
  

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]


            iter_num += 1

           
            progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1]))
                                      
                                           ])                      

            if iter_num == epoch_length: #at the end of every epoch 
            
                
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
            


                if C.verbose:
                   
                   
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))
                    elapsed_time = (time.time()-start_time)/60

                curr_loss = loss_rpn_cls + loss_rpn_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(
                            best_loss, curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights(C.model_path)
                new_row = {'loss_rpn_cls': round(loss_rpn_cls, 3), 'loss_rpn_regr': round(loss_rpn_regr, 3), 'curr_loss': round(curr_loss, 3), 'elapsed_time': round(elapsed_time, 3), 'mAP': 0}
                record_df = record_df.append(
                    new_row, ignore_index=True)
                record_df.to_csv(C.record_path, index=0)

                break

        except Exception as e:
            print('Exception: {}'.format(e))
            continue

print('Training complete, exiting.')


