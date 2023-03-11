using DelimitedFiles
using Random

Random.seed!(19981021)

sigmoid(z::Real) = one(z) / (one(z) + exp(-z))
sigmoid_d(z::Real) = sigmoid(z) * (1 - sigmoid(z))

mse(yₚ, yₜ) = (yₚ - yₜ)^2


function load_data(data_path)
	xy = readdlm(data_path, ',', Int, '\n' )
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

	epoch = 0
	cnt = size(xs)[1] - 1

	while epoch < max_epoch
		epoch += 1
        e = 0.0
		for step in 0:batch_size:cnt
			dw₁ = zeros(Float64, size(w₁)...)
			dw₂ = zeros(Float64, size(w₂)...)
			db₁ = zeros(Float64, size(b₁)...)
			db₂ = zeros(Float64, size(b₂)...)
			
			t0 = time()
			@inbounds @simd for i in 1:batch_size
				x = xs[step+i, :]
				y = ys[step+i, :]

				z₁ = w₁ * x + b₁
				a₁ = sigmoid.(z₁)
				z₂ = w₂ * a₁ + b₂
				a₂ = sigmoid.(z₂)

				e = sum(mse.(a₂, y)) / 10

				dw₂ += 2/10 * (a₂ - y) .* sigmoid_d.(z₂) .* transpose(a₁)
				db₂ += 2/10 * (a₂ - y) .* sigmoid_d.(z₂)

				da₁ = transpose(sum(2 * (a₂ - y) .* sigmoid_d.(z₂) .* w₂, dims = 1))

				dw₁ += da₁ .* sigmoid.(z₁) .* transpose(x)
				db₁ += da₁ .* sigmoid.(z₁)
			end

			t1 = time()

			w₁ = w₁ - lr .* dw₁ ./ batch_size
			w₂ = w₂ - lr .* dw₂ ./ batch_size
			b₁ = b₁ - lr .* db₁ ./ batch_size
			b₂ = b₂ - lr .* db₂ ./ batch_size

			time_cost = Int(t1-t0)
            @show step, e, time_cost
		end
        @show epoch, e
	end

    w₁, b₁, w₂, b₂
end


function forward(xs, ys, w₁, b₁, w₂, b₂)
    res = 0
    cnt = size(xs)[1]
    for i = 1:cnt
        yt = sigmoid.(w₂ * sigmoid.(w₁ * xs[i,:] + b₁) + b₂)
        res += (argmax(yt)[1] == argmax(ys[i,:]))
    end
    @show res / cnt
end



function main()
	xs, ys = load_data("backpropagation/data/mnist_train.csv")
	w₁, b₁, w₂, b₂ = network(xs, ys, 1e-4, 100, 2)

    xt, yt = load_data("backpropagation/data/mnist_test.csv")
    forward(xt, yt, w₁, b₁, w₂, b₂)
end

main()


#=
(epoch, e) = (3, 0.13428515255534304)
res / cnt = 0.8124
=#