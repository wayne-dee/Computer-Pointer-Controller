# Computer Pointer Controller

*TODO:* Write a short introduction to your project

=>Computer controller pointer is a mouse controller app model that move the mouse pointer using
  the head postion face land marks and eye gaze
=> Computer pointer controller use the intergration of three models eye gaze, facial landmark and face detection

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.
**1** Step 1
    Install the openvino toolkit version 2020.1 and above you can find the installation guides
    here 
    https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html
    and also you can find the documentations here

    https://docs.openvinotoolkit.org/latest/index.html

**2** Step 2
    clone my github repository to your project folder

    -set environment variables for openvion on on your machines for windows
    cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
    setupvars.bat

    Then cd to you project folder.
    -install the requirements in the folder the requirements.txt by this command
    pip install -r requirements.txt

**3** Step 3
    make directory models from your working directory
    mkdir models
    cd models

    Then Download the following models from the model downloader
    cd cd C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools/tools/model_downloader
1. Face Detection Model
    ```
     python ./downloader.py --name face-detection-adas-binary-0001 --output_dir C:\your_project_folder\models\

    ```

2. Facial Landmarks Detection Model**
    ```
     python ./downloader.py --name landmarks-regression-retail-0009 --output_dir C:\your_project_folder\models\ 

    ```
3. Head Pose Estimation Model**
    ```
    python ./downloader.py --name head-pose-estimation-adas-0001 --output_dir C:\your_project_folder\models\

    ```
4. Gaze Estimation Model
    ```
     python ./downloader.py --name gaze-estimation-adas-0002 --output_dir C:\your_project_folder\models\

    ```
**4** Model structure
    computer-pointer-controller  
    |
    |--bin
    |   |--demo.mp4
    |--README.md
    |--requirements.txt
    --models // though ignored
    |--src
        |--face_detection.py
        |--face_landmarks_detection.py
        |--gaze_estimation.py
        |--head_pose_estimation.py
        |--input_feeder.py
        |--app.py
        |--mouse_controller.py
    ```
### Demo
*TODO:* Explain how to run a basic demo of your model.
 **1**
    ```
    cd src - the folder in your working directoty
    ```
    run the following command
    ```
        python app.py -fd ..\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 -fl ..\models\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009 -hp ..\models\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001 -ge ..\models\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002 -i ..\bin\demo.mp4 -d CPU 
     ```
     you can include the following flags for visualization
     ```
     python app.py -fd ..\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 -fl ..\models\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009 -hp ..\models\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001 -ge ..\models\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002 -i ..\bin\demo.mp4 -d CPU -flags fd he ge
     ```

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.
    **1 Commands arguments**
    1. -fd => face_detection command path to face detection .xml and .bin modele file
    2. -fl => path to .xml and .bin file of facial landmarks model
    3. -hp => path to .xml and .bin file  of head pose model
    4. -ge => path to .xml and .bin files of gaze estimation model
    5. -i => path to input file eg CAM or video file
    6. -d => targeted device ie CPU, GPU, FPGA and MYRIAD
    7. -flags => viualization for the mention models by adding 

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.
**Running the models with FP32**
    1. Results
    Face Detection Load Time: time: 499.806 ms
    Facial landmarks load Time: time: 295.276 ms
    Head pose load time: time: 413.455 ms
    Gaze estimation load time: time: 436.634 ms
    Total loading time taken: time: 1645.172 ms
    All models are loaded successfully..
    Feeder is loaded
    Time counter: 19.000 seconds
    Total inference time: 28.760 seconds
    FPs: 0.661 fps
**Running Model on FP16**
    2. results
    Face Detection Load Time: time: 432.395 ms
    Facial landmarks load Time: time: 142.149 ms
    Head pose load time: time: 175.163 ms
    Gaze estimation load time: time: 208.498 ms
    Total loading time taken: time: 959.205 ms
    All models are loaded successfully..
    Feeder is loaded
    Time counter: 10.000 seconds
    Total inference time: 14.964 seconds
    FPs: 0.668 fps
## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.
    **Results**
    - As you see model loading time reduces as shown on face detection reduces almost 60 ms
    - Facial landmarks reduces from 295.276 ms to 142.149
    - Head pose model reduces from 413.455 to 175.163
    - as also for gaze estimation

**Inference time**
    The ingerence time for FP32 is 28.760 which is much higher than for FP16 which is 14.964 seconds

**Frames per second**
 Fps increases from FP32 of 0.661 fps to 0.668fps in FP16

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
