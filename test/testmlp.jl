using KnetOnnx

#file_path = "/Users/egeersu/Desktop/KnetOnnx/src/PosterDemo/mlp.onnx"
#file_path = "/Users/egeersu/Desktop/KnetOnnx/src/recurrent.onnx"
file_path = "/Users/egeersu/Desktop/KnetOnnx/src/PosterDemo/cnn.onnx"
graph = ONNXtoGraph(file_path)
PrintGraph(graph)

model = KnetModel(file_path)
