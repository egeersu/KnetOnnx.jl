using KnetOnnx

file_path = "/Users/egeersu/Desktop/KnetOnnx/src/recurrent.onnx";
graph = ONNXtoGraph(file_path)
PrintGraph(graph)
model = KnetModel(file_path)
println(model.tensors)
