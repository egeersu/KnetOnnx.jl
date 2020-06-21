# CONSTANT
mutable struct constant_layer
    data
end

(l::constant_layer)() = l.data

# Shape
mutable struct shape_layer
end

(l::shape_layer)(x) = [size(x)...]

# SQUEEZE
mutable struct squeeze_layer
    axes
end

function (s::squeeze_layer)(x)
    new_size = []
    for (i, dim) in enumerate(size(x))
        if dim>1; push!(new_size, dim); end
    end
    new_size = (new_size...,)
    reshape(x, new_size)
end

# UN-SQUEEZE
mutable struct unsqueeze_layer
    axes
end

function (u::unsqueeze_layer)(x)
    data = [t for t in size(x)]
    axes = [a+1 for a in u.axes]
    for i in axes; insert!(data, i, 1); end
    new_size = (data...,)
    reshape(x, new_size)
end


# Gather
struct Gather
    axis #increment before adding (julia 0->1)
end

function (g::Gather)(data, indices)
    indices_size = size(indices)

    indices = (x->(x+1)).(indices) # increment for Julia
    indices = (x->(Int32(x))).(indices) # set floats to Int for bug-free indexing

    if length(indices_size) == 1
        return gather_rank1(data, indices)
    end
    if length(indices_size) == 2
        return gather_rank2(data, indices)
    end
    if length(indices_size) > 2
        print("Gather for indices with rank > 2 are not implemented yet.")
    end
end


function gather_rank1(data, indices)
    new_data = []
    axis1 = size(indices)[1]

    for a1 in (1:axis1)
        current_index = indices[a1]
        #get_data = data[:,current_index]
        get_data = data[:,current_index]
        push!(new_data, get_data)
    end
    new_data
end

function gather_rank2(data, indices)
    new_data = []
    axis1, axis2 = size(indices)

    for a1 in (1:axis1)
        mini_list = []
        for a2 in (1:axis2)
            current_index = indices[a1,a2]
            get_data = data[:,current_index]
            push!(mini_list, get_data)
        end
        push!(new_data, mini_list)
    end
    new_data
end

#Concat
mutable struct Concat
    axis
end

function (l::Concat)(args...)
    if l.axis == 0; return vcat(args...);
    else; return hcat(args...); end
end

# Constant of Shape
struct ConstantOfShape
     value
end

function (c::ConstantOfShape)(input)
    output = fill(c.value, input...)
end


# converts weird onnx params to knet params (need this to train)
function ConvertParams(w)
    out = zeros(size(w))
    size_w = size(w)

    if length(size_w) == 1;
        for i in 1:size(w)[1]; out[i] = w[i]; end; end

    if length(size_w) == 2
        for i in 1:size(w)[1]; for j in 1:size(w)[2]; out[i,j] = w[i,j]; end; end; end;

    if length(size_w) == 3
        for i in 1:size(w)[1]; for j in 1:size(w)[2]; for z in 1:size(w)[3]; out[i,j,z] = w[i,j,z]; end; end; end; end;

    if length(size_w) == 4
        for i in 1:size(w)[1]; for j in 1:size(w)[2]; for z in 1:size(w)[3]; for k in 1:size(w)[4]; out[i,j,z,k] = w[i,j,z,k]; end; end; end; end; end;

    if length(size_w) == 5
        for i in 1:size(w)[1]; for j in 1:size(w)[2]; for z in 1:size(w)[3]; for k in 1:size(w)[4]; for t in 1:size(w)[5]; out[i,j,z,k,t] = w[i,j,z,k,t]; end; end; end; end; end; end;

    Knet.param(out)
end
