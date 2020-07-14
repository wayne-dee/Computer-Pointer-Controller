'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IENetwork,IECore

class Landmarks_Detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.device = device
        self.extensions = extensions
        self.model_weights = model + '.bin'
        self.model_structure = model + ".xml"

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        try:
            self.core = IECore()
            self.network = IENetwork(model=self.model_structure, weights=self.model_weights)
                    
            self.infer_net = self.core.load_network(network=self.network, device_name=self.device,num_requests=1)
            
            self.input_name = next(iter(self.network.inputs))
            self.input_shape = self.network.inputs[self.input_name].shape
            self.output_names = next(iter(self.network.outputs))
        except:
            print('error on load model gaze est')
    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        try:
            processed_frame = self.preprocess_input(image.copy())

            input_dict = {self.input_name:processed_frame}
            infer_output = self.infer_net.infer(input_dict)
            coords = self.preprocess_output(infer_output, image)

            height=image.shape[0]
            width=image.shape[1]
            coords = coords* np.array([width, height, width, height])
            coords = coords.astype(np.int32)
            ## 10 is the value of eye area  sorounding
            # right eye
            re_xmin=coords[2]-10
            re_ymin=coords[3]-10
            re_xmax=coords[2]+10
            re_ymax=coords[3]+10
            # left eye
            le_xmin=coords[0]-10
            le_ymin=coords[1]-10
            le_xmax=coords[0]+10
            le_ymax=coords[1]+10
            
            left_eye =  image[le_ymin:le_ymax, le_xmin:le_xmax]
            right_eye = image[re_ymin:re_ymax, re_xmin:re_xmax]
            eye_coords = [[le_xmin,le_ymin,le_xmax,le_ymax], [re_xmin,re_ymin,re_xmax,re_ymax]]

            return left_eye, right_eye, eye_coords
        except:
            print('Error predict head pose')

    def check_model(self):
        try:
            supported = self.core.query_network(network=self.network, device=self.device)
            unsupported = [
                layer for layer in self.network.layers.keys() 
                if layer not in supported
            ]
            if len(unsupported) > 0:
                print("unsupported layers found:" + str(unsupported))
                exit(1)
            print("All layers are supported !!!")
        except:
            print('Error on check model land marks estimation')

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        #  shape: [1x3x48x48]
        try:
            frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            frame = frame.transpose(2, 0, 1)
            frame_processed = frame.reshape(1, *frame.shape)

            return frame_processed
        except:
            print('Error on preprocess input land marks')

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        try:
            box = outputs[self.output_names][0]

            leye_x = box[0].tolist()[0][0]
            leye_y = box[1].tolist()[0][0]
            reye_x = box[2].tolist()[0][0]
            reye_y = box[3].tolist()[0][0]
            
            return (leye_x, leye_y, reye_x, reye_y)
        except:
            print('Error on preprocess output land marks')
