using Statistics
using BenchmarkTools

function my_rand(d...)
    rand(Float32, d...) .* 2 .- 1
end

function normalize(arr)
    (arr .- mean(arr)) / std(arr)
end

function softmax(x)
    tmp = exp.(x)
    tmp ./ sum(tmp)
end
struct EncoderLayer
    HEAD_COUNT::Int
    TOKEN_COUNT::Int
    TOKEN_DIM::Int

    WQ::Array{Float32}
    WK::Array{Float32}
    WV::Array{Float32}

    W::Array{Float32}
    WF::Array{Float32}

    EncoderLayer(HEAD_COUNT::Int, TOKEN_COUNT::Int, TOKEN_DIM::Int) = new(
        HEAD_COUNT,
        TOKEN_COUNT,
        TOKEN_DIM,
        ATTEN_DIM,
        my_rand(HEAD_COUNT, TOKEN_DIM, TOKEN_DIM//HEAD_COUNT),
        my_rand(HEAD_COUNT, TOKEN_DIM, ATTEN_DIM//HEAD_COUNT),
        my_rand(HEAD_COUNT, TOKEN_DIM, ATTEN_DIM//HEAD_COUNT),
        my_rand(TOKEN_DIM, TOKEN_DIM),
        my_rand(TOKEN_DIM, TOKEN_DIM)
    )
end

Base.show(io::IO, layer::EncoderLayer) = print(io, "EncoderLayer(head_count=$(layer.HEAD_COUNT), token_count=$(layer.TOKEN_COUNT), token_dim=$(layer.TOKEN_DIM), atten_dim=$(layer.ATTEN_DIM))")



function forward(layer::EncoderLayer, tokens::Array{Float32})
    Z = zeros(layer.TOKEN_COUNT, layer.ATTEN_DIM * layer.HEAD_COUNT)
    @simd for h in 1:layer.HEAD_COUNT
        WQ = layer.WQ[h, :, :]
        WK = layer.WK[h, :, :]
        WV = layer.WV[h, :, :]
        Q = tokens * WQ
        K = tokens * WK
        V = tokens * WV
        attention_output = Q * transpose(K) ./ sqrt(layer.TOKEN_DIM) |> softmax
        attention_output = attention_output * V
        Z[:, ((h-1)*layer.ATTEN_DIM+1):h*layer.ATTEN_DIM] = attention_output
    end
    W = layer.W
    multi_attention_output = Z * W

    add_output = multi_attention_output + tokens
    nomalized_output = add_output |> normalize

    WF = layer.WF
    ffn_output = nomalized_output * WF
    add_output = ffn_output + nomalized_output
    normalized_output = add_output |> normalize

    return normalized_output
end

tokens = my_rand(1024, 4096)
layer = EncoderLayer(16, 1024, 4096)

@benchmark forward($layer, $tokens)