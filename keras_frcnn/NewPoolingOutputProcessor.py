from keras.engine.topology import Layer
from keras_frcnn.vgg_occlusion_net import OcclusionNet
from keras_frcnn import thresholding_helper
import keras.backend as K
import numpy as np

if K.backend() == 'tensorflow':
    import tensorflow as tf


class NewPoolingOutputProcessor(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        x=[pooling_output,occlusion_mask]
        pooling_output= (1,None,7,7,512)
        occlusion_mask=(1,None,7,7,2)
    '''
    def __init__(self,pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois


        super(NewPoolingOutputProcessor, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, 512, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, 512
  
    #format x=[pooling_output,occlusion_mask]
    #pooling_output and occlusion_mask are of the same dimension 
    #pooling_output =(1,10,7,7,512)
    #occlusion_mask =(1,10,7,7,1) =>zeros and ones 
    def call(self, x,mask=None):
        
        # #unpack variables
        pooling_output=x[0]
        occlusion_mask=x[1]
        # # print("Pooling: ", K.int_shape(pooling_output))
        # # print("Mask: ",K.int_shape(occlusion_mask))
        # print("Pooling: ", pooling_output)
        # print("Mask: ",occlusion_mask)
        # x=pooling_output
        zeros=K.zeros_like(pooling_output)

        #modify pooling_output based on the mask 

        #drop out all of the spatial locations that correspond to the 1's in the mask  
        output=K.switch(occlusion_mask,zeros,pooling_output) 

        # print("Output: ",K.int_shape(output))
      


      

      

        return output
    
    

    def get_config(self):
            config = {'pool_size': self.pool_size,
                    'num_rois': self.num_rois}
            base_config = super(NewPoolingOutputProcessor, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))