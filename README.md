<img src="https://github.com/onnx/onnx/blob/master/docs/ONNX_logo_main.png?raw=true" width="400">

KnetOnnx reads an ONNX file and creates the corresponding Model in Knet that can be re-designed, re-trained or simply used for inference.

If you are planning to move your models from PyTorch or Tensorflow to Knet, or simply desiring to play with popular pre-trained neural networks: KnetOnnx provides that functionality.

[Open Neural Network Exchange (ONNX)](https://onnx.ai/)
 is a community project created by Facebook and Microsoft. It provides a definition of an extensible computation graph model, as well as definitions of built-in operators and standard data types.

Operators are implemented externally to the graph, but the set of built-in operators are portable across frameworks. Every framework supporting ONNX will provide implementations of these operators on the applicable data types.

Although not all operations are implemented yet, visit ONNX's [model zoo](https://github.com/onnx/models) to download pre-trained, state-of-the-art models in the ONNX format.

Once you download the ONNX file, call KnetModel() with the ONNX file's path to create the model.

## Tutorial

Here is how you create the Knet model corresponding to an ONNX file and perform a forward pass:

```julia
using KnetOnnx

#provide the ONNX file's path
model = KnetModel("vgg16.onnx");

#dummy input for prediction
x = ones(224,224,3,10)

#call KnetModel object with the model input
model(x) #the output is a 1000×10 Array{Float32,2}
```

Check out our [tutorial notebooks](http://localhost:8888/tree/test/tutorials) to learn how you can transfer your models from PyTorch to Knet: 
> [MLP](https://github.com/egeersu/KnetOnnx.jl/blob/master/test/tutorials/Knet_MLP.ipynb) - [CNN](https://github.com/egeersu/KnetOnnx.jl/blob/master/test/tutorials/Knet_CNN.ipynb) - [VGG16])(https://github.com/egeersu/KnetOnnx.jl/blob/master/test/tutorials/VGG.ipynb)

## Supported Operations
- [x] ReLU
- [x] LeakyReLU
- [x] Conv
- [x] MaxPool
- [x] Dropout
- [x] Flatten
- [x] Gemm
- [x] Add
- [x] BatchNormalization
- [x] ImageScaler
- [x] RNN
- [x] Unsqueeze
- [x] Squeeze
- [x] Concatenate
- [x] ConstantOfShape
- [x] Shape
- [x] Constant

## Collaboration
Here are some cool ideas if you want to collaborate:
- Export functionality. This could be done in one of 3 ways:

	- (1) By only exporting models that are impelemented as a [KnetModel](https://github.com/egeersu/KnetOnnx.jl/blob/master/src/KnetModel.jl). These models have to use [KnetLayers](https://github.com/egeersu/KnetOnnx.jl/tree/master/src/KnetLayers), so one could implement functions that convert KnetLayers into strings. These strings would then be combined to construct the model.onnx file. The structure of the model (inputs, outputs, connections) can be inferred from the KnetModel.

	- (2) By running a dummy input through the model and then collecting the Julia operations. These more primitive operations could then be turned into strings and combined according to the order of operations. This would be a completely novel project and does not depend on anything implemented by this package.

	- (3) A hybrid approach. Use (1) for KnetModels & functions that make use of KnetLayers. Use (2) if the layers/operations being used are unknown. 

- Adding a new [KnetLayer](https://github.com/egeersu/KnetOnnx.jl/tree/master/src/KnetLayers)
- Adding a new [Converter](https://github.com/egeersu/KnetOnnx.jl/blob/master/src/converters.jl) ([ONNX Operator](https://github.com/onnx/onnx/blob/master/docs/Operators.md) -> [KnetLayer](https://github.com/egeersu/KnetOnnx.jl/tree/master/src/KnetLayers))
- Downloading & testing ONNX models from the [zoo](https://github.com/onnx/models) and sending bug reports.
- Trying to import your models from [ONNX-complete frameworks.](https://onnx.ai/supported-tools.html#buildModel) 
- Writing tests for [KnetLayers](https://github.com/egeersu/KnetOnnx.jl/tree/master/src/KnetLayers) and [Converters.](https://github.com/egeersu/KnetOnnx.jl/blob/master/src/converters.jl)
- Adding Type Constraints to converters. (See [Onnx Operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md) for more info)
- [ONNX RUNTIME](https://microsoft.github.io/onnxruntime/) (advanced)

> If you want to better understand the structure of this package, please read our [Technical Report](https://github.com/egeersu/KnetOnnx.jl/blob/master/KnetOnnx-Report.pdf).
