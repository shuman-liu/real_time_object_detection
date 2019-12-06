# real_time_object_detection
A programs that can detect object in real time video stream.

## Author
Shuman Liu

## Model
The model I use is the TensorFlow model called Tensorflow detection model zoo.<br>
the link to the model: [click here](https://github.com/tensorflow/models/blob/477ed41e7e4e8a8443bc633846eb01e2182dc68a/object_detection/g3doc/detection_model_zoo.md)<br>
this detection model is pre-trained on the COCO dataset.<br>
the link to the dataset: [click here](http://cocodataset.org/#home)

## Requirement
1. platform Windows, Mac (Linux virtual machine may have problem with opening the camera)
1. Python 3.7
2. Package to install
  - tensorflow
  - opencv-python
  - pillow
  - matplotlib
  - numpy
  
 ## How to use
 Just run the ob_dect.py script and the program will up and running. And it will use the default camera on your computer.
