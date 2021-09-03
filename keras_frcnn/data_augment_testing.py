#given an image, randomly select a A X A square on it to be occluded  
import numpy as np 
from keras_frcnn.random_eraser import get_random_eraser_v1, get_random_eraser_v2
from skimage.util import random_noise

# def img_augment_occlusion(img,x):
#     (height,width)=img[:,:,0].shape
#     map=np.zeros((height,width))

#     #randomly select elements to drop out based on the mode 
  

 
#     num=height*width
#     # print("Total: ",num)
#     selected_num=int(num*x)
#     selected_indices=np.random.choice(num,selected_num,replace=False)
#     # print("Selected: ",selected_indices)

#     for index in selected_indices:
#         row=index//width
#         col=index%width 

#         map[row,col]=1 
    

#     img[map==1]=0 

#     return img 


def img_augment_occlusion_v1(img,p=0.5,pixel_level=False):
    """Given an image, randomly occlude a rectangular part of it"""

    eraser=get_random_eraser_v1(p=p,pixel_level=pixel_level)
    return eraser(img)

def img_augmented_occlusion_v2(img,xmin,xmax,ymin,ymax,p=0.5,pixel_level=False):
    """Given an image and an ROI, randomly occlude a part of the ROI"""

    eraser=get_random_eraser_v2(p=p,pixel_level=pixel_level)
    return eraser(img,xmin,xmax,ymin,ymax)

def img_augment_noise(img,noise_type="s&p",amount=0.05):
    noise_img=random_noise(img,mode=noise_type,amount=amount)
    noise_img = np.array(255*noise_img, dtype = 'uint8')

    return noise_img


