struct Tensor{T<:Real}
    data::Array{T}
    grad::Array{T}
    parent::Set{Tensor{T}}
    _op::String
    _backward::Union{Nothing,Function}
    Tensor{T}(data::Array{T}) where {T<:Real} = new(data, zeros(T, size(data)), Set{Tensor{T}}(), "", Nothing)
    Tensor{T}(data::Array{T}, parent::Set{Tensor{T}}, op::String) where {T<:Real} = new(data, zeros(T, size(data)), parent, op, Nothing)
end

function Base.:+(a::Tensor{T}, b::Tensor{T}) where {T<:Real}
    @assert size(a.data) == size(b.data)
    out = Tensor{T}(a.data + b.data, Set{Tensor{T}}([a, b]), "+")
    function _backward()
        a.grad += b.grad .* out.grad
        b.grad += a.grad .* out.grad
        nothing
    end
    out._backward = _backward
    out
end


function Base.:*(a::T, b::Tensor{T}) where {T<:Real}
    out = Tensor{T}(a * b.data, Set{Tensor{T}}([b]), "$a*")
    function _backward()
        b.grad += a * out.grad
        nothing
    end
    out._backward = _backward
    out
end

function Base.:*(a::Tensor{T}, b::Tensor{T}) where {T<:Real}
    @assert size(a.data)[2] == size(b.data)[1]
    out = Tensor{T}(a.data * b.data, Set{Tensor{T}}([a, b]); op="*")
    function _backward()
        a.grad += b.data' * out.grad
        b.grad += a.data' * out.grad
        nothing
    end
    out._backward = _backward
    out
end



function Base.show(io::IO, t::Tensor{T}) where {T<:Real}
    println(io, "Tensor{$T}(")
    println(io, "    data: ", t.data)
    println(io, "    grad: ", t.grad)
    println(io, ")")
end


w1 = Tensor{Float32}(randn(Float32, 8, 4))
b1 = Tensor{Float32}(randn(Float32, 8))
x = Tensor{Float32}(randn(Float32, 4))
a1 = w1 * x + b1
@show a1

w2 = Tensor{Float32}(randn(Float32, 4, 8))
b2 = Tensor{Float32}(randn(Float32, 4))
a2 = w2 * a1 + b2
@show a2

w3 = Tensor{Float32}(randn(Float32, 1, 4))
b3 = Tensor{Float32}(randn(Float32, 1))
a3 = w3 * a2 + b3
@show a3