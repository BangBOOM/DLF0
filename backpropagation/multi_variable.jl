#=
y = σ(w₂ ̇σ(w₁x + b₁) + b₂)

We just simplify the question, given a fix x = [0.7, 0.3, 0.4, 0.2] and y = [0.1, 0.9], 
use the rule chain to detect the value of w₁, w₂, b₁, b₂
=#

using Random
Random.seed!(1024)

sigmoid(z::Real) = one(z) / (one(z) + exp(-z))
mse(yₚ, yₜ) = (yₚ - yₜ)^2
sigmoid_d(z::Real) = sigmoid(z) * (1 - sigmoid(z))

x = [0.7, 0.3, 0.4, 0.2]
y = [0.1, 0.9]

w₁ = rand(3, 4)
b₁ = rand(3)

w₂ = rand(2, 3)
b₂ = rand(2)

# z₁ = w₁ * x + b₁
# a₁ = sigmoid.(z₁)

# z₂ = w₂ * a₁ + b₂
# a₂ = sigmoid.(z₂)

# e = sum(mse.(a₂, y))

# @show e

lr = 0.01
step = 0
while step <= 100
    global step, w₁, b₁, w₂, b₂
    step += 1

    z₁ = w₁ * x + b₁
    a₁ = sigmoid.(z₁)
    z₂ = w₂ * a₁ + b₂
    a₂ = sigmoid.(z₂)

    e = sum(mse.(a₂, y))

    dw₂ = 2 * (a₂ - y) .* sigmoid_d.(z₂) .* transpose(a₁)
    db₂ = 2 * (a₂ - y) .* sigmoid_d.(z₂)

    da₁ = transpose(sum(2 * (a₂ - y) .* sigmoid_d.(z₂) .* w₂, dims=1))

    dw₁ = da₁ .* sigmoid.(z₁) .* transpose(x)
    db₁ = da₁ .* sigmoid.(z₁)

    w₁ = w₁ - lr .* dw₁
    w₂ = w₂ - lr .* dw₂
    b₁ = b₁ - lr .* db₁
    b₂ = b₂ - lr .* db₂

    if step % 10 == 0
        @show db₂
        @show size(db₂)
        @show step, e
    end
end 