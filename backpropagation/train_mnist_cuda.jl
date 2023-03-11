using CUDA

@inline sigmoid(z::Real) = one(z) / (one(z) + exp(-z))

function sigmoid_d!(z)
    i = gridDim().x
    z[i] = sigmoid(z[i])
    return nothing
end


x = rand(700)
x_b = CUDA.rand(700, 300)

w = rand(700, 300)
b = rand(300)

w_b = CUDA.rand(700, 300)
b_b = CUDA.rand(300)

@cuda threads=length(b) sigmoid_d!(b_b)