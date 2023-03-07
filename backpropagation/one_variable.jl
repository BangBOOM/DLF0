#=
y = σ(w₂ ̇σ(w₁x + b₁) + b₂)

We just simplify the question, given a fix x = 0.3 and y = 0.7, 
use the rule chain to detect the value of w₁, w₂, b₁, b₂

=#

using Random

Random.seed!(1024)

sigmoid(z::Real) = one(z) / (one(z) + exp(-z))
mse(yₚ, yₜ) = (yₚ - yₜ)^2
sigmoid_d(z::Real) = sigmoid(z) * (1 - sigmoid(z))


x = 0.3
y = 0.1

w₁ = rand()
b₁ = rand()
w₂ = rand()
b₂ = rand()

# z₁ = w₁ * x + b₁
# a₁ = sigmoid(z₁)

# z₂ = w₂ * a₁ + b₂
# a₂ = sigmoid(z₂)

# e = mse(a₂, y)

# r = 0
# @show r, e

lr = 0.01
r = 0
while r <= 5000
    global r, w₁, b₁, w₂, b₂
    r += 1
    z₁ = w₁ * x + b₁
    a₁ = sigmoid(z₁)

    z₂ = w₂ * a₁ + b₂
    a₂ = sigmoid(z₂)

    e = mse(a₂, y)

    dw₂ = 2 * (a₂ - y) * sigmoid_d(z₂) * a₁
    db₂ = 2 * (a₂ - y) * sigmoid_d(z₂)

    da₁ = 2 * (a₂ - y) * sigmoid_d(z₂) * w₂

    dw₁ = da₁ * sigmoid(z₁) * x
    db₁ = da₁ * sigmoid(z₁)
    
    if r % 1000 == 0
        @show w₁, b₁, w₂, b₂
        # @show dw₁, db₁, dw₂, db₂
        @show r, e
    end

    w₁ = w₁ - lr * dw₁
    w₂ = w₂ - lr * w₂
    b₁ = b₁ - lr * db₁
    b₂ = b₂ - lr * db₂
end
