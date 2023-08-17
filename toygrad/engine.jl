mutable struct Tensor{T<:Real}
    data::Array{T}
    grad::Array{T}
    parent::Set{Tensor{T}}
    _op::String
    _backward::Function
    Tensor{T}(data::Array{T}) where {T<:Real} = new(data, zeros(T, size(data)), Set{Tensor{T}}(), "", _ -> nothing)
    Tensor{T}(data::Array{T}, parent::Set{Tensor{T}}, op::String) where {T<:Real} = new(data, zeros(T, size(data)), parent, op, _ -> nothing)
end

zero_grad!(t::Tensor{T}) where {T<:Real} = t.grad .= zero(T)
one_grad!(t::Tensor{T}) where {T<:Real} = t.grad .= one(T)

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
    out = Tensor{T}(a.data * b.data, Set{Tensor{T}}([a, b]), "*")
    function _backward()
        a.grad += out.grad * b.data'
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


x = Tensor{Float32}([1.0f0 2.0f0; 3.0f0 4.0f0])
y = Tensor{Float32}([2.0f0; 3.0f0])
z = Tensor{Float32}([-2.0f0 1.0f0])

o1 = x * y
o2 = z * o1
@show o2

one_grad!(o2)
o2._backward()
@show z

o1._backward()
@show x
@show y