import tflite2onnx

tflite_file = "hand_landmark_lite.tflite"
onnx_file = "hand_landmark_lite.onnx"

tflite2onnx.convert(tflite_file, onnx_file)