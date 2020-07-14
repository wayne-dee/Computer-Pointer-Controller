'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IENetwork,IECore

class Face_Detection:
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
            
            # self.check_model()
            self.infer_network = self.core.load_network(network=self.network, device_name=self.device,num_requests=1)
            self.input_name = next(iter(self.network.inputs))
            self.input_shape = self.network.inputs[self.input_name].shape
            self.output_name = next(iter(self.network.outputs))
        except:
            print('Error occurred while loading model at face detection')
    def predict(self, image, conf_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        try:
            frame_processed = self.preprocess_input(image)
            input_dict = {self.input_name:frame_processed}

            network_output = self.infer_network.infer(input_dict)
            box = self.preprocess_output(network_output, conf_threshold)
            if (len(box)==0):
                return 0, 0
            ## takes first image
            box = box[0] 
            height=image.shape[0]
            width=image.shape[1]
            box = box* np.array([width, height, width, height])
            box = box.astype(np.int32)
            
            crop_face = image[box[1]:box[3], box[0]:box[2]]
            return crop_face, box
        except:
            print('Error ocurred while at predictt on face detection')

    def check_model(self):
        ## check unsuported layers
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
            print('Error on check model face detection estimation')


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        #  [1x3x384x672] - An input image in the format [BxCxHxW]
        try:    
            frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            frame = frame.transpose(2, 0, 1)
            frame_processed = frame.reshape(1, *frame.shape)

            return frame_processed
        except:
            print('Error ocurred while preprocessing input on face detection')
    def preprocess_output(self, outputs, conf_threshold):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        try:
            coordinates =[]
            outputs = outputs[self.output_name][0][0]
            for box in outputs:
                conf = box[2]
                if conf>conf_threshold:
                    x_min=box[3]
                    y_min=box[4]
                    x_max=box[5]
                    y_max=box[6]
                    coordinates.append([x_min,y_min,x_max,y_max])
            return coordinates
        except:
            print('Error ocurred while preprocessing output on face detection')
        
