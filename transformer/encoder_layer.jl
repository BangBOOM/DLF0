# I have a sentence: I am learning Julia. 
# I will init a matrix with 4 rows and 1024 columns to represent the sentence.
using Statistics


function my_rand(d1, d2)
    rand(d1, d2).* 2 .- 1
end

function normalize(arr)
    
    (arr .- mean(arr)) / std(arr)
end

function softmax(x)
    tmp = exp.(x)
    tmp ./ sum(tmp)
end

function positional_encoding(pos, dim, b)
    pos_encoding = zeros(pos, dim)
    @simd for p in 0:pos-1
        for i in 0:div(dim, 2)-1
            tmp = b ^ (2i/dim)
            pos_encoding[p+1, 2i+1] = sin(p/tmp)
            pos_encoding[p+1, 2i+2] = cos(p/tmp)
        end
    end
    pos_encoding
end


TOKEN_CNT = 4
TOKEN_DIM = 1024
ATTEN_DIM = 512

tokens = my_rand(TOKEN_CNT, TOKEN_DIM)

# first step: do positional encoding
tokens += positional_encoding(TOKEN_CNT, TOKEN_DIM, 1_000)


# enter Self-Attention
# one head
#=
WQ = my_rand(TOKEN_DIM, ATTEN_DIM)
WK = my_rand(TOKEN_DIM, ATTEN_DIM)
WV = my_rand(TOKEN_DIM, ATTEN_DIM)

Q = tokens * WQ
K = tokens * WK
V = tokens * WV

attention_output = Q * transpose(K) / sqrt(TOKEN_DIM) |> softmax 
attention_output = attention_output * V
=#

# multi head
HEAD_CNT = 8

Z = rand(TOKEN_CNT, ATTEN_DIM * HEAD_CNT)
for h in 1:HEAD_CNT
    WQ = my_rand(TOKEN_DIM, ATTEN_DIM)
    WK = my_rand(TOKEN_DIM, ATTEN_DIM)
    WV = my_rand(TOKEN_DIM, ATTEN_DIM)
    Q = tokens * WQ
    K = tokens * WK
    V = tokens * WV
    attention_output = Q * transpose(K) / sqrt(TOKEN_DIM) |> softmax 
    attention_output = attention_output * V
    Z[:, ((h-1)*ATTEN_DIM+1):h*ATTEN_DIM] = attention_output
end

W = my_rand(ATTEN_DIM * HEAD_CNT, TOKEN_DIM)
multi_attention_output = Z * W

# ADD & Normalization
add_output = multi_attention_output + tokens
normlized_output = add_output |> normalize

WF = my_rand(TOKEN_DIM, TOKEN_DIM)
ffn_output = normlized_output * WF

add_output = ffn_output + tokens
normlized_output = add_output |> normalize



