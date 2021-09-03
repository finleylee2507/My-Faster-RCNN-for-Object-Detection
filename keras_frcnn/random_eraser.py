#citation: https://github.com/yu4u/cutout-random-erasing 

import numpy as np


def get_random_eraser_v1(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    

    def eraser(input_img):
        """Given an image, randomly occlude a rectangular part of it"""

        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = np.random.rand()

        if p_1 > p: #skip the augmentation 
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w #area? 
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r)) #width of occluded area 
            h = int(np.sqrt(s * r)) #height of occluded area 
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h: #if the seleted w and h satisfy the constraint, break out and set values 
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h) #pixel value for occluded area

        input_img[top:top + h, left:left + w] = c

        return input_img

    return eraser


def get_random_eraser_v2(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):

    def eraser(input_img,xmin,xmax,ymin,ymax):
        """Given an image and an ROI, randomly occlude a part of the ROI"""
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape
        roi_width=xmax-xmin
        roi_height=ymax-ymin
 
        p_1 = np.random.rand()

        if p_1 > p: #skip the augmentation 
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * roi_height * roi_width #area? 
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r)) #width of occluded area 
            h = int(np.sqrt(s * r)) #height of occluded area 
            left = np.random.randint(xmin, xmax)
            top = np.random.randint(ymin, ymax)
            # print("X min: ",xmin,"X max: ",xmax, "Y min: ",ymin,"Y max: ",ymax)
            # #print("Image width, ",img_w," image height: ",img_h)
            # print("Roi width: ",roi_width, "roi height: ",roi_height)
            # print("W: ",w,"h: ",h)
            # print("Left: ",left, "Top: ",top)
            if left + w <= xmax and top + h <= ymax: #if the seleted w and h satisfy the constraint, break out and set values 
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h) #pixel value for occluded area

        input_img[top:top + h, left:left + w] = c

        return input_img 

    return eraser