
haarcascade_frontalface_Path = "./models/OpenCV/haarcascade_frontalface_default.xml" # for cascade method

# 8 bit Quantized version using Tensorflow ( 2.7 MB )
TFmodelFile = "./models/OpenCV/tensorflow/TF_faceDetectorModel_uint8.pb"
TFconfigFile = "./models/OpenCV/tensorflow/TF_faceDetectorConfig.pbtxt"


# FP16 version of the original caffe implementation ( 5.4 MB )
CAFFEmodelFile = "./models/OpenCV/caffe/FaceDetect_Res10_300x300_ssd_iter_140000_fp16.caffemodel"
CAFFEconfigFile = "./models/OpenCV/caffe/FaceDetect_Deploy.prototxt"


dlib_mmod_model_Path = "./models/Dlib/mmod_human_face_detector.dat" # for Dlib MMOD 

dlib_5_face_landmarks_path = './models/Dlib/shape_predictor_5_face_landmarks.dat'

ultra_light_640_onnx_model_path = './models/UltraLight/ultra_light_640.onnx'
ultra_light_320_onnx_model_path = './models/UltraLight/ultra_light_320.onnx'


yoloV3FacePretrainedWeightsPath = './models/Yolo/yolo_weights/yolov3-wider_16000.weights'
yoloV3FaceConfigFilePath = './models/Yolo/yolo_models_config/yolov3-face.cfg'
yoloV3FaceClassesFilePath = './models/Yolo/yolo_labels/face_classes.txt'
