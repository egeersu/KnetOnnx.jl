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