mlp_path = "/Users/egeersu/Desktop/KnetOnnx/src/PosterDemo/mlp.onnx"
graph = ONNXtoGraph(mlp_path)
PrintGraph(graph)

mlp = KnetModel(mlp_path)
