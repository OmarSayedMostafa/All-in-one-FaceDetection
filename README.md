# All-in-one-FaceDetection
this is a collected repos for most famous algorithms and models for face detection collocted from github repos and structed in more organised way for speed and accuracy comparison.

* each enviroment specify which model would work best for it from speed and accuracy prespective

## this repos contains:
- Cloned from **[learnopencv website](https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/)** // **[github](https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison)**
	Highly recommend to visit the website so you will know more about how the algorithms/model work and for further more understanding, also it contains a wide material and tutorial on Computer Vision tasks.
	- **OpenCV** 

		- [Opencv classisc haarcascade](https://github.com/OmarSayedMostafa/All-in-one-FaceDetection/blob/master/openCVHaarCascade.py).
		- [Opencv DNN](https://github.com/OmarSayedMostafa/All-in-one-FaceDetection/blob/master/OpenCVDnn.py)
			* Single-Shot-Multibox detector SSD and uses ResNet-10 Architecture as backbone.
				1. [FloatPoint16 version of the original caffe implementation ( 5.4 MB )](https://github.com/OmarSayedMostafa/All-in-one-FaceDetection/tree/master/models/OpenCV/caffe)
       	 	2. [8 bit Quantized version using Tensorflow ( 2.7 MB )](https://github.com/OmarSayedMostafa/All-in-one-FaceDetection/tree/master/models/OpenCV/tensorflow)
	- **Dlib**
		1. [Dlib MMOD DNN : Maximum-Margin Object Detector ( MMOD ) with CNN based features.](https://github.com/OmarSayedMostafa/All-in-one-FaceDetection/blob/master/dlibFaceDetection.py)
		2. [Dlib HOG Detector: Based on HoG features and SVM](https://github.com/OmarSayedMostafa/All-in-one-FaceDetection/blob/master/dlibFaceDetection.py)

- **[MTCNN](https://github.com/OmarSayedMostafa/All-in-one-FaceDetection/blob/master/mtcnnFaceDetector.py)** [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)

- **[face_recognition](https://github.com/OmarSayedMostafa/All-in-one-FaceDetection/blob/master/faceRecogintion.py)** [orginal repo](https://github.com/ageitgey/face_recognition)  Built using dlib's state-of-the-art face recognition built with deep learning. The model has an accuracy of 99.38% on the Labeled Faces in the Wild benchmark.

- **[YoloV3 From DarkNet](https://github.com/OmarSayedMostafa/All-in-one-FaceDetection/blob/master/yoloV3FaceDetection.py)** cloned from these **[github repo](https://github.com/fyr91/face_detection.git)** and **[github repo](https://github.com/sthanhng/yoloface)**

- **[UltraLightOnnxFaceDetector](https://github.com/OmarSayedMostafa/All-in-one-FaceDetection/blob/master/ultra_light_Detector.py)** by Linzaer and MobileFaceNet, 
	- the most accurate and fast worked for me on cpu, cloned from **[Medium blog post](https://towardsdatascience.com/real-time-face-recognition-with-cpu-983d35cc3ec5)** // **[Github Repo](//github.com/fyr91/face_detection.git)** by [Author: fyr91]
