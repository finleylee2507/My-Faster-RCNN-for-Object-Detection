from keras.engine.topology import Layer
from keras_frcnn.vgg_occlusion_net import OcclusionNet
import keras_frcnn.thresholding_helper 
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf

class PoolingOutputProcessor(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois
        self.occlusion_net=OcclusionNet(input_height=7, input_width=7, input_channel=512)

        super(PoolingOutputProcessor, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x,occlusion_path,thresholding_option, mask=None):
        #initialize the occlusion net 
        occlusion=self.occlusion_net 
        print("loading occlusion model from ", occlusion_path)
        occlusion.load_weights(occlusion_path, by_name=True)
        occlusion.compile(optimizer='sgd',loss='mae')

				#print("Here!")
        for i in range(0,self.num_rois): #iterate over all the ROIs 
            temp_sample=x[:,i]
            occlusion_prediction=occlusion.predict_on_batch(temp_sample)

            #apply occlusion on the pooling layer output according to the selected policy 
            if(thresholding_option=='direct'): #apply direct thresolding 
                for row in range(occlusion_prediction.shape[0]):
                    for col in range(occlusion_prediction.shape[1]):
                        if (occlusion_prediction[0,row,col,0]>=0.5): #drop out 
                            x[0,i,row,col,:]=0 
    


            if(thresholding_option=='sampling'): #apply sampling thresholding 
                # print("Prediction: ",occlusion_prediction[0,:,:,0])
                processed_output=thresholding_helper.threshold_by_sampling(occlusion_prediction[0,:,:,0],1/3,1/2) #could tweak the parameters 
                # print("Processed output: ", processed_output) 

                for row in range(0,processed_output.shape[0]):
                    for col in range(0,processed_output.shape[1]):
        
                        if(processed_output[row,col]==1):
                            x[0,i,row,col,:]=0 #drop out all channels in the corresponding spatial location 
                            
                        


            if(thresholding_option=='random'): #apply random thresolding 
                processed_output=thresholding_helper.random_thresholding(occlusion_prediction[0,:,:,0],0.3) #could tweak the parameters 

                for row in range(0,processed_output.shape[0]):
                    for col in range(0,processed_output.shape[1]):
        
                        if(processed_output[row,col]==1):
                            x[0,i,row,col,:]=0 #drop out all channels in the corresponding spatial location 
                                                    


      

        return x
    
    
    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(PoolingOutputProcessor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
