module onnx_utils

# Converter Utils
function UInt8toFloat32(val)
    dims = val.dims
    data = val.raw_data
    indices = collect(1:4:length(data))
    data_list = []
    for i in indices; push!(data_list, float_from_bitstring(Float32, generate_bitstring(data[i:i+3]))); end;
    if length(val.dims) != 0
        new_size = tuple(val.dims...)
        data_list = reshape(data_list, new_size)
    end
    #cast to Float32 (migth adjust this later)
    return Float32.(data_list)
end

function generate_bitstring(raw_4)
    bits = ""
    for i in reverse(raw_4); bits *= bitstring(i); end
    bits
end

function float_from_bitstring(::Type{T}, str::String) where {T<:Base.IEEEFloat}
    unsignedbits = Meta.parse(string("0b", str))
    thefloat  = reinterpret(T, unsignedbits)
    return thefloat
end

export UInt8toFloat32
end
