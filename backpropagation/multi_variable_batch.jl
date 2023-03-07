#=
y = σ(w₂ ̇σ(w₁x + b₁) + b₂)

We just simplify the question, given a fix x = [0.7, 0.3, 0.4, 0.2] and y = [0.1, 0.9], 
use the rule chain to detect the value of w₁, w₂, b₁, b₂
=#
using Statistics
using Random
Random.seed!(1024)

sigmoid(z::Real) = one(z) / (one(z) + exp(-z))
mse(yₚ, yₜ) = (yₚ - yₜ)^2
sigmoid_d(z::Real) = sigmoid(z) * (1 - sigmoid(z))

batch_size = 5
x = [0.7 0.3 0.4 0.2; 0.2 0.7 0.4 0.3; 0.2 0.1 0.3 0.3; 0.1 0.7 0.5 0.3; 0.9 0.7 0.4 0.3]
y = [0.1 0.9; 0.5 0.8; 0.3 0.8; 0.3 0.4; 0.3 0.1]

x = transpose(x)
y = transpose(y)

w₁ = rand(3, 4)
b₁ = rand(3)

w₂ = rand(2, 3)
b₂ = rand(2)

z₁ = w₁ * x .+ b₁
a₁ = sigmoid.(z₁)

z₂ = w₂ * a₁ .+ b₂
a₂ = sigmoid.(z₂)

@show a₂
e = sum(mse.(a₂, y), dims=1)
@show e

db₂ = 2 * (a₂ - y) .* sigmoid_d.(z₂)
dw₂ = zeros(2, 3, batch_size)
@inbounds @simd for i in 1:batch_size
    dw₂[:, :, i] = db₂[:, i] .* transpose(a₁[:, i])
end

da₁ = zeros(3, batch_size)
@inbounds @simd for i in 1:batch_size
    da₁[:, i] .= sum(db₂[:, i] .* w₂)
end

@assert size(da₁) == size(a₁)

db₁ = da₁ .* sigmoid.(z₁)
dw₁ = zeros(size(w₁)...,batch_size)
@inbounds @simd for i in 1:batch_size
    dw₁[:, :, i] = db₁[:, i] .* transpose(x[:, i])
end

lr = 0.01

w₁ = w₁ - lr .* mean(dw₁, dims=3)
w₂ = w₂ - lr .* mean(dw₂, dims=3)
b₁ = b₁ - lr .* mean(db₁, dims=2)[:]
b₂ = b₂ - lr .* mean(db₂, dims=2)[:]

@show size(w₁)
@show size(b₁)
# @show db₂
# @show a₁
# @show sum(db₂, dims=1)


# step = 0
# while step <= 10000
#     global step, w₁, b₁, w₂, b₂
#     step += 1

#     z₁ = w₁ * x + b₁
#     a₁ = sigmoid.(z₁)
#     z₂ = w₂ * a₁ + b₂
#     a₂ = sigmoid.(z₂)

#     e = sum(mse.(a₂, y))

#     dw₂ = 2 * (a₂ - y) .* sigmoid_d.(z₂) .* transpose(a₁)
#     db₂ = 2 * (a₂ - y) .* sigmoid_d.(z₂)

#     da₁ = transpose(sum(2 * (a₂ - y) .* sigmoid_d.(z₂) .* w₂, dims=1))

#     dw₁ = da₁ .* sigmoid.(z₁) .* transpose(x)
#     db₁ = da₁ .* sigmoid.(z₁)

#     w₁ = w₁ - lr .* dw₁
#     w₂ = w₂ - lr .* dw₂
#     b₁ = b₁ - lr .* db₁
#     b₂ = b₂ - lr .* db₂

#     if step % 1000 == 0
#         @show step, e
#     end
# end 