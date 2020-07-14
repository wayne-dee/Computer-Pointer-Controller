'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IENetwork,IECore

class Head_Pose:
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
            frame_processed = self.preprocess_input(image)
            input_dict = {self.input_name:frame_processed}

            network_out = self.infer_net.infer(input_dict)

            return self.preprocess_output(network_out)
        except:
            print('Error on predict head pose')
        
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
            print('Error on check model gaze estimation')

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # shape: [1x3x60x60] - An input image in [1xCxHxW] format. Expected color order is BGR.
        try:
            frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            frame = frame.transpose(2, 0, 1)
            frame_processed = frame.reshape(1, *frame.shape)

            return frame_processed
        except:
            print('Error preprocess input head pose')

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        try:
            # convert the outputs into an array
            coordinates = np.array([
                outputs['angle_y_fc'].tolist()[0][0],
                outputs['angle_p_fc'].tolist()[0][0],
                outputs['angle_r_fc'].tolist()[0][0]
            ])
        
            return coordinates
        except:
            print('Error on preprocess output head pose')

