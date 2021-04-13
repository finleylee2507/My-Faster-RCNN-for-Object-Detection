import cv2
import numpy as np
import copy
import random
import skimage  

#https://towardsdatascience.com/complete-image-augmentation-in-opencv-31a6b02694f5 


# def channel_shift(img, value):
#     value = int(random.uniform(-value, value))
#     img = img + value
#     img[:,:,:][img[:,:,:]>255]  = 255
#     img[:,:,:][img[:,:,:]<0]  = 0
#     img = img.astype(np.uint8)
#     return img

def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def gaussianBlur(img,ksize=11):
	return cv2.GaussianBlur(img, (ksize,ksize),0)


def randomNoise(img): #needs to be fixed 
	img=skimage.util.random_noise(img,mode='gaussian')
	img = np.array(255*img, dtype = 'uint8')
	return img 




def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data) #copy over the image to perform augmentation 

	img = cv2.imread(img_data_aug['filepath'])

	count=len(config.list) 

	if augment:
		rows, cols = img.shape[:2]

		#randomly select an option 
		lucky_number=np.random.randint(0,count) 
		lucky_option=config.list[lucky_number]


		#for the horizontal flips 
		if lucky_option =='horizontal_flips':
			#print('horizontal')
			img = cv2.flip(img, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2 

		#for the vertical flips 
		if lucky_option =='vertical_flips':
			#print('vertical')
			img = cv2.flip(img, 0)
			for bbox in img_data_aug['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2


		#for the random noise 
		if lucky_option =='noise':
			#print('noise')
			img=randomNoise(img)
		
		#for the gaussian blur
		if lucky_option =='blur':
			#print('blur')
			img=gaussianBlur(img)
		
		#for the brightness 
		if lucky_option =='brightness':
			#print('brightness')
			img=brightness(img,0.5,1.5) 
		
		#for the rotational 
		if lucky_option =='rotational':
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 0)
			elif angle == 180:
				img = cv2.flip(img, -1)
			elif angle == 90:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 1)
			elif angle == 0:
				pass

			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1
				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2        
				elif angle == 0:
					pass
			
		if lucky_option =='original': #use the original image 
			pass 

		
			
		
		
			

		# #for the channel shift
		# if config.channel_shift :#and np.random.randint(0,2) ==0:
		# 	print('shift')
		# 	img=channel_shift(img,60)
		
		


	img_data_aug['width'] = img.shape[1]
	img_data_aug['height'] = img.shape[0]
	return img_data_aug, img
