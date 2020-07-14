import numpy as np
import logging as log
import os
import sys
import time
from head_pose_estimation import Head_Pose
from face_detection import Face_Detection
from gaze_estimation import Gaze_Estimation
from facial_landmarks_detection import Landmarks_Detection
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder
import cv2

# python app.py -fd ..\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 -fl ..\models\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009 -hp ..\models\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001 -ge ..\models\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002 -i ..\bin\demo.mp4 -d CPU  -flags fd he ge

try:
    def build_argparser():

        parser = ArgumentParser()
        parser.add_argument("-fd", "--face_detection", required=True, type=str,
                            help="Path to .xml file of Pretrained model face detection model.")
        parser.add_argument("-fl", "--face_landmark", required=True, type=str,
                            help="Path to .xml file of pretrained facial landmark detection model.")
        parser.add_argument("-hp", "--head_pose", required=True, type=str,
                            help="Path to .xml file of pretrained head pose estimation model.")
        parser.add_argument("-ge", "--gaze_estimation", required=True, type=str,
                            help="Path to .xml file of pretrained model gaze estimation model.")
        parser.add_argument("-i", "--input", required=True, type=str,
                            help="Path to camera cam or video file")
        parser.add_argument("-flags", "--mode_visualization", required=False, nargs='+', default=[],
                            help="You will need tospecify the flags separated by space ie fd fl hp ge" )
        parser.add_argument("-l", "--cpu_extension", required=False, type=str, default=None,
                            help="targeted custom layers if have any or to shared libraries.")
        parser.add_argument("-conf", "--conf_threshold", required=False, type=float, default=0.6,
                            help="Confidence threshold for frame detection helps to improve accuracy")
        parser.add_argument("-d", "--device", type=str, default="CPU",
                            help="These are devices you are targeting to run the mode ie CPU, MYRIAD,FPGA,VPU ")
        
        return parser

    def main(args):
        ## loading models
        try:
            input_file = args.input
            mode_visualization = args.mode_visualization
            
            if input_file == "CAM":
                input_feeder = InputFeeder("cam")
            else:
                if not os.path.isfile(input_file):
                    log.error("ERROR: INPUT PATH IS NOT VALID")
                    exit(1)
                input_feeder = InputFeeder("video", input_file)

            face_detection_class = Face_Detection(
                model=args.face_detection, 
                device=args.device, 
                extensions=args.cpu_extension
            )
            face_landmarks_class = Landmarks_Detection(
                model=args.face_landmark,
                device=args.device, 
                extensions=args.cpu_extension
            )
            head_pose_class = Head_Pose(
                model=args.head_pose, 
                device=args.device, 
                extensions=args.cpu_extension
            )
            gaze_estimation_class = Gaze_Estimation(
                model=args.gaze_estimation, 
                device=args.device, 
                extensions=args.cpu_extension
            )
        
            mouse_control = MouseController('medium', 'fast')
            start_time = time.time()

            ## Load the models one by one and all necessary info

            face_det_time = time.time()
            face_detection_class.load_model()
            print("Face Detection Load Time: time: {:.3f} ms".format((time.time() - face_det_time) * 1000))

            face_land_time = time.time()
            face_landmarks_class.load_model()
            print("Facial landmarks load Time: time: {:.3f} ms".format((time.time() - face_land_time) * 1000))

            head_po_time = time.time()
            head_pose_class.load_model()
            print("Head pose load time: time: {:.3f} ms".format((time.time() - head_po_time) * 1000))

            gaze_est_time = time.time()
            gaze_estimation_class.load_model()
            print("Gaze estimation load time: time: {:.3f} ms".format((time.time() - gaze_est_time) * 1000))

            total_time = time.time() - start_time
            print("Total loading time taken: time: {:.3f} ms".format(total_time * 1000))

            print("All models are loaded successfully..")

            input_feeder.load_data()
            print("Feeder is loaded")
        except:
            print('Error occured on loading models in app')

        ## performing inferences
        try:
            start_inference_time = time.time()
            frame_count = 0
            for flag, frame in input_feeder.next_batch():
                if not flag:
                    break
                frame_count+=1
                if frame_count==0:
                    cv2.imshow('video',cv2.resize(frame,(700,700)))
            
                key = cv2.waitKey(60)
                crop_face, face_coords = face_detection_class.predict(frame.copy(), args.conf_threshold)
                if type(crop_face)==int:
                    log.error("Unable to detect the face.")
                    if key==27:
                        break
                    continue
                
                ## perform inference
                head_angle = head_pose_class.predict(crop_face.copy())
                left_eye, right_eye, eye_coords = face_landmarks_class.predict(crop_face.copy())
                mouse_position, gaze_vector = gaze_estimation_class.predict(left_eye, right_eye, head_angle)
                
                ## checking for extra flags
                if (not len(mode_visualization)==0):
                    p_frame = frame.copy()
                    if ('fd' in mode_visualization):
                        p_frame = crop_face
                    if ('fl' in mode_visualization):
                        cv2.rectangle(
                            crop_face, 
                            (eye_coords[0][0]-10, eye_coords[0][1]-10), 
                            (eye_coords[0][2]+10, eye_coords[0][3]+10), 
                            (0,255,0), 1)
                        cv2.rectangle(
                            crop_face, 
                            (eye_coords[1][0]-10, eye_coords[1][1]-10), 
                            (eye_coords[1][2]+10, eye_coords[1][3]+10), 
                            (0,255,0,), 1)
                        
                    if ('hp' in mode_visualization):
                        cv2.putText(
                            p_frame, 
                            "Head Positions: :{:.2f} :{:.2f} :{:.2f}".format(
                                head_angle[0],head_angle[1],head_angle[2]), (10, 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)

                    if ('ge' in mode_visualization):
                        i, j, k = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160

                        l_eye =cv2.line(
                            left_eye.copy(), 
                            (i-k, j-k), (i+k, j+k), 
                            (0,255,255), 2
                            )
                        cv2.line(
                            l_eye, 
                            (i-k, j+k), 
                            (i+k, j-k), 
                            (255,0,255), 2
                            )

                        r_eye = cv2.line(
                            right_eye.copy(), 
                            (i-k, j-k), 
                            (i+k, j+k), 
                            (0,255,255), 2
                            )
                        cv2.line(
                            r_eye, 
                            (i-k, j+k), 
                            (i+k, j-k), 
                            (0,255,255), 2
                            )

                        l_eye = crop_face[eye_coords[0][1]:eye_coords[0][3],eye_coords[0][0]:eye_coords[0][2]]
                        r_eye = crop_face[eye_coords[1][1]:eye_coords[1][3],eye_coords[1][0]:eye_coords[1][2]]
                        
                    cv2.imshow("visual for client",cv2.resize(p_frame,(700,700)))
                
                if frame_count%1==0:
                    mouse_control.move(mouse_position[0],mouse_position[1])    
                if key==27:
                        break
            ## working on inference time and frames per second
            total_infer_time = time.time() - start_inference_time
            frames_per_sec = int(frame_count) / total_infer_time

            print("Time counter: {:.3f} seconds".format(frame_count))
            print("Total inference time: {:.3f} seconds".format(total_infer_time))
            print("FPs: {:.3f} fps ".format(frames_per_sec))
        except:
            print('Error on performing inference in app file')

        print("All Done...")

        cv2.destroyAllWindows()
        input_feeder.close()

    if __name__ == '__main__':
        args = build_argparser().parse_args()

        main(args)

except KeyboardInterrupt:
    sys.exit()
 

