using CUDA
using Test


function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # stride = gridDim().x * blockDim().x

    # gdx = gridDim().x
    # bdx = blockDim().x
    # bix = blockIdx().x
    # tix = threadIdx().x

    # @cuprintln("gdx: $gdx, bdx:$bdx, bix:$bix, tix:$tix, index:$index")
    # for i = index:stride:length(y)
    #     @inbounds y[i] += x[i]
    # end
    @inbounds y[index] += x[index]
    return
end


N = 256

y_d = CUDA.fill(1.0f0, N)
x_d = CUDA.fill(2.0f0, N)

numblocks = ceil(Int, N/8)

@cuda threads=8 blocks=numblocks gpu_add3!(y_d, x_d)
# synchronize()
@test all(Array(y_d) .== 3.0f0)