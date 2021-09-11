import os
from shutil import copyfile
import random
from progress.bar import Bar
from optparse import OptionParser


# the method randomly selects k images from the original VOC directory and copies over the corresponding annotations files
def create_smaller_voc(k):

    original_img_dir = './NEW_TRAIN/VOCdevkit/VOC2012/JPEGImages'
    original_annotation_dir = './NEW_TRAIN/VOCdevkit/VOC2012/Annotations'
    original_train_val_dir = './NEW_TRAIN/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'

    # create new directories and files
    # if not os.path.exists('../TRAIN_EXPERIMENT'):
    # 	os.makedirs('../TRAIN_EXPERIMENT')
    if not os.path.exists('./TRAIN_EXPERIMENT/VOCdevkit/VOC2012/Annotations'):
        os.makedirs('./TRAIN_EXPERIMENT/VOCdevkit/VOC2012/Annotations')
    if not os.path.exists('../TRAIN_EXPERIMENT/VOCdevkit/VOC2012/JPEGImages'):
        os.makedirs('./TRAIN_EXPERIMENT/VOCdevkit/VOC2012/JPEGImages')
    if not os.path.exists('../TRAIN_EXPERIMENT/VOCdevkit/VOC2012/ImageSets/Main'):
        os.makedirs('./TRAIN_EXPERIMENT/VOCdevkit/VOC2012/ImageSets/Main')

    f = open('./TRAIN_EXPERIMENT/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt', "w")

    counter = 0
    # iterate over the image folder
    with Bar('Processing...', max=k) as bar:

        for filename in os.listdir(original_img_dir):
            # print(filename)
            annotation_name = filename[:-3]
            annotation_name += 'xml'
            # print(annotation_name)
            train_val_dir_filename = filename[:-4]

            if os.path.exists(os.path.join(original_annotation_dir, annotation_name)) and random.random() >= 0.5 and counter < k:
                # copy annotation file over
                copyfile(os.path.join(original_annotation_dir, annotation_name), os.path.join(
                    './TRAIN_EXPERIMENT/VOCdevkit/VOC2012/Annotations', annotation_name))

                # copy image over
                copyfile(os.path.join(original_img_dir, filename), os.path.join(
                    './TRAIN_EXPERIMENT/VOCdevkit/VOC2012/JPEGImages', filename))

                # write the filename to the new .txt file
                line_to_write = train_val_dir_filename+"\n"
                f.write(line_to_write)

                counter += 1
                bar.next()


# #call the method
# create_smaller_voc(k=5000)

if __name__ == "__main__":
	parser = OptionParser()
	parser.add_option("-k", "--k",  type="int",dest="set_size",
					help="Size of the data set.")

	(options, args) = parser.parse_args()

	k = options.set_size

	create_smaller_voc(k)
