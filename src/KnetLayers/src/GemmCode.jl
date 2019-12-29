#Gemm Layer

mutable struct Gemm 
    weight
    bias
    alpha
    beta
end

function Gemm(;alpha, beta=1, transA=0, transB=0, w, b)
    if transA != 0; w = transpose(w); end
    if transB != 0; x = transpose()
    Gemm(w,b,alpha,beta)
end
    
function (g::Gemm)(x)
    g.alpha * g.weight * x + (g.beta * g.bias)    
end
    
  

#graph node to Gemm Layer
function node_to_gemm(node, weightdims, g)
    alpha = node.attribute[:alpha]
    beta = node.attribute[:beta]
    transA = node.attribute[:transA]
    transB = node.attribute[:transB]
    transB = 0
    w_name = node.input[2]
    b_name = node.input[3]
    #wsize = weightdims[w_name]
    #bsize = weightdims[b_name]
    w = g.initializer[w_name]
    b = g.initializer[b_name]
    println(transA, transB)
    layer = Gemm(alpha=alpha, beta=beta, transA=transA, transB=transB, w=w, b=b)
end
