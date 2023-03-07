using Statistics
using DelimitedFiles
using Random

Random.seed!(19981021)

sigmoid(z::Real) = one(z) / (one(z) + exp(-z))
sigmoid_d(z::Real) = sigmoid(z) * (1 - sigmoid(z))

mse(yₚ, yₜ) = (yₚ - yₜ)^2


function load_data(data_path)
	xy = readdlm(data_path, ',', Int, '\n')
	sample_cnt = size(xy)[1]
	y = zeros(Float64, size(xy)[1], 10)
	@inbounds @simd for i in 1:sample_cnt
		y[i, xy[i, 1]+1] = 1.0
	end
	x = xy[:, 2:end]
	x, y
end

function network(xs, ys, lr, batch_size, max_epoch)
	w₁ = rand(300, 784) - rand(300, 784)
	b₁ = rand(300) - rand(300)
	w₂ = rand(10, 300) - rand(10, 300)
	b₂ = rand(10) - rand(10)

	i = 10
	step = 1
    epoch = 0
    cnt = size(xs)[1] - 1
    while epoch < max_epoch
        epoch += 1
        e = 0.0
        for step in 0:batch_size:cnt
            x = transpose(xs[step+1:step+batch_size, :])
            y = transpose(ys[step+1:step+batch_size, :])

            z₁ = w₁ * x .+ b₁
            
            a₁ = sigmoid.(z₁)
            z₂ = w₂ * a₁ .+ b₂
            a₂ = sigmoid.(z₂)
        
            e = sum(mse.(a₂, y), dims = 1) / 10
        
            db₂ = 2 / 10 * (a₂ - y) .* sigmoid_d.(z₂)
            dw₂ = zeros(size(w₂)..., batch_size)
            @simd for i in 1:batch_size
                dw₂[:, :, i] = db₂[:, i] .* transpose(a₁[:, i])
            end
        
            da₁ = zeros(300, batch_size)
            @inbounds @simd for i in 1:batch_size
                da₁[:, i] .= sum(db₂[:, i] .* w₂)
            end
        
            @assert size(da₁) == size(a₁)
        
            db₁ = da₁ .* sigmoid.(z₁)
            dw₁ = zeros(size(w₁)...,batch_size)
            @inbounds @simd for i in 1:batch_size
                dw₁[:, :, i] = db₁[:, i] .* transpose(x[:, i])
            end
        
            w₁ = w₁ - lr .* mean(dw₁, dims=3)[:,:,1]
            w₂ = w₂ - lr .* mean(dw₂, dims=3)[:,:,1]
            b₁ = b₁ - lr .* mean(db₁, dims=2)[:]
            b₂ = b₂ - lr .* mean(db₂, dims=2)[:]
        end
        @show epoch, e[1]
    end

    w₁, b₁, w₂, b₂
end


function main()
	xs, ys = load_data("backpropagation/data/mnist_train.csv")
	w₁, b₁, w₂, b₂ = network(xs, ys, 0.001, 100, 2)
end

main()