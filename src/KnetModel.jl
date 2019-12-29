#import Knet

"""
    KnetModel

    * tensors: A dictionary. Given the tensor name as a string, returns the actual tensor.

    * model_layers: returns the list of layers (the actual layers themselves)

    * model_inputs: returns the list of inputs (the names of the tensors)

    * model_outputs: returns the list of outputs (the names of the outputs)

"""
mutable struct KnetModel
    tensors #dictionary: string -> tensor
    model_layers
    model_inputs
    model_outputs
    graph
end

"""
Given a Graph, construct the corresponding KnetModel
"""
function KnetModel(g::KnetOnnx.Types.Graph)
    model_layers = get_ModelLayers(g)
    tensors = TensorDict(model_layers, g)
    model_inputs = [i.name for i in g.input]
    model_outputs = [o.name for o in g.output]
    graph = g
    KnetModel(tensors, model_layers, model_inputs, model_outputs, graph)
end

function KnetModel(path::String)
    KnetModel(ONNXtoGraph(path))
end


"""
    TensorDict
    Initializes KnetModel.tensors by putting the names in as keys, but the values are Nothing.
"""
#omitted weights, might need them later
function TensorDict(model_layers, g)
    tensors = Dict()
    for layer in model_layers
        for input in layer.inputs; tensors[input] = Nothing; end
        for input in layer.outputs; tensors[input] = Nothing; end
    end
    for t in keys(g.initializer)
        if t in keys(tensors); tensors[t] = g.initializer[t]; end
    end
    tensors
end

"""
    ModelLayer
        * inputs: a list of tensor names.
          These tensors will be used for forward calculation of the layer.
        * outputs: a list of tensor names.
          The outputs of the forward calculation will be saved to Model.tensors under these keys.
        * layer: a Knet Layer.
          If you are constructing your own ModelLayer make sure the number of inputs and outputs matches the functionality of the KnetLayer you are using.

"""
mutable struct ModelLayer
    inputs #list of strings
    layer # a KnetLayer
    outputs #list of strings
end

function ModelLayer(node, g)
    (args, layer, outputs) = convert(node, g)
    ModelLayer(args, layer, outputs)
end

function get_ModelLayers(g)
    ModelLayers = []
    for node in g.node; push!(ModelLayers, ModelLayer(node, g)); end
    return ModelLayers
end


# FORWARD CALCULATIONS
function forward(km::KnetModel, ml::ModelLayer)

        # GATHER INPUTS
    for input in ml.inputs
        if km.tensors[input] == Nothing; return "oops!"; end
    end

        # FORWARD PASS
        # if only one input is requried, pass the first element
        # if more than one input is required, pass all elements
        # simply check the length of requried inputs for the model
    inputs = (key-> km.tensors[key]).(ml.inputs)
    if length(inputs) == 1; out = ml.layer(inputs[1]);
        else; out = ml.layer(inputs...); end
        #else; out = ml.layer(inputs[1]); end

        # SAVE OUTPUTS
        # check if there are multiple outputs (rnn etc.) before saving them to model.tensors
    if length(ml.outputs) == 1; km.tensors[ml.outputs[1]] = out;
        else; for output in ml.outputs; km.tensors[output] = out; end; end
 end

function (m::KnetModel)(args...)

    # reset model.tensors (figure out a smarter/faster reset)
    m.tensors = TensorDict(m.model_layers, m.graph)

    # REGISTER X
    # dumb version
    # check if we want multiple inputs (x should be a list) or a single input (x is a single array)
    #if length(m.model_inputs) == 1; m.tensors[m.model_inputs[1]] = x;
        #else; for (i,model_input) in enumerate(m.model_inputs); m.tensors[model_input] = x[i]; end; end

    for (i, arg) in enumerate(args)
        m.tensors[m.model_inputs[i]] = arg
    end


    #m.tensors[m.model_inputs...] = x

    # LOOP UNTIL ALL TENSORS ARE CALCULATED
    # do until all model.tensors are filled
    # iterate over all layers and call forward on that layer
    while Nothing in values(m.tensors)
        for layer in m.model_layers
            forward(m, layer)
        end
    end

    # Gather outputs
    outs = []
    #m.tensors[m.model_outputs...]
    # DUMB VERSION
    # could be multiple
    if length(m.model_outputs) == 1; outs = m.tensors[m.model_outputs[1]];
        else; for out in m.model_outputs; push!(outs, m.tensors[out]); end;  end

    # return outputs
    return outs

end

"""
    PrintModelTensors(models::KnetModel)
    Displays your model.tensor, showing the size of the tensors that are calculated.
    Might be useful for debugging your model.
"""
function PrintModelTensors(model::KnetModel)
    tensors = model.tensors
    for k in keys(tensors)
        if tensors[k] == Nothing; println(k, "\t=> ", "Nothing")
        else println(k, "\t=> ", size(tensors[k])); end
    end
end

import Knet: Data

(m::KnetModel)(x,y) = Knet.nll(m(x), y)

"""
    Calling KnetModel with a Knet.Data object computes the mean nll for
"""
(m::KnetModel)(d::Data) = Knet.mean(m(x,y) for (x,y) in d)
