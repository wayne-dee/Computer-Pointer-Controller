'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
import math

from openvino.inference_engine import IENetwork,IECore

class Gaze_Estimation:
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
    def predict(self, left_eye, right_eye, head_position):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        try:
            left_e = self.preprocess_input(left_eye)
            right_e = self.preprocess_input(right_eye)
            self.infer_net.start_async(
                request_id=0,inputs={
                    'left_eye_image': left_e,
                    'right_eye_image': right_e,
                    'head_pose_angles': head_position
                    }
                )

            if self.infer_net.requests[0].wait(-1) == 0:
                output = self.infer_net.requests[0].outputs[self.output_names]
                coords = self.preprocess_output(output[0], head_position)

                return output[0], coords
        except:
             print('Error on predict gaze estimation')
        
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
        try:
            input_shape = self.network.inputs['right_eye_image'].shape
            frame = cv2.resize(image, (input_shape[3], input_shape[2]))
            frame = frame.transpose(2, 0, 1)
            processed_frame = frame.reshape(1, *frame.shape)

            return processed_frame
        except:
            print('Error on preprocess input gaze estimation')
    def preprocess_output(self, outputs, head_position):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        try:
            roll = head_position[2]
            gaze_vector = outputs / cv2.norm(outputs)

            cosineValue = math.cos(roll * math.pi / 180.0)
            sineValue = math.sin(roll * math.pi / 180.0)


            x_value = gaze_vector[0] * cosineValue * gaze_vector[1] * sineValue
            y_value = gaze_vector[0] * sineValue * gaze_vector[1] * cosineValue
            return (x_value, y_value)
        except:
            print('Error occured on preprocess output gaze estimation')