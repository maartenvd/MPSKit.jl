"""
    LazySum{O} <: AbstractVector{O}

Type that represents a lazy sum i.e explicit summation is only done when needed. 
This type is basically an `AbstractVector` with some extra functionality to calculate things efficiently.

## Fields
- ops -- Vector of summable objects

---

## Constructors
    LazySum(x::Vector)

"""
struct LazySum{O} <: AbstractVector{O}
    ops::Vector{O}
end

# for the AbstractArray interface
Base.size(x::LazySum) = size(x.ops)
Base.getindex(x::LazySum, i) = x.ops[i]
Base.length(x::LazySum) = prod(size(x))
Base.similar(x::LazySum, ::Type{S}, dims::Dims) where {S} = LazySum(similar(x.ops, S, dims))
Base.setindex!(A::LazySum, X, i) = setindex!(A.ops, X, i)

# Holy traits
TimeDependence(x::LazySum) = istimed(x) ? TimeDependent() : NotTimeDependent()
istimed(x::LazySum) = any(istimed, x)

# constructors
LazySum(x) = LazySum([x])
LazySum(ops::AbstractVector, fs::AbstractVector) = LazySum(map(MultipliedOperator, ops, fs))

# wrapper around _eval_at
safe_eval(::TimeDependent, x::LazySum, t::Number) = map(O -> _eval_at(O, t), x)
safe_eval(::NotTimeDependent, x::LazySum) = sum(_eval_at, x)

# For users
# using (t) should return NotTimeDependent LazySum
(x::LazySum)(t::Number) = safe_eval(x, t)
Base.sum(x::LazySum) = safe_eval(x) #so it works for untimedoperator

# we define the addition for LazySum and we do the rest with this
function Base.:+(SumOfOps1::LazySum, SumOfOps2::LazySum)
    return LazySum([SumOfOps1..., SumOfOps2...])
end

Base.:+(op1::LazySum, op2) = op1 + LazySum(op2)
Base.:+(op1, op2::LazySum) = LazySum(op1) + op2
Base.:+(op1::MultipliedOperator, op2::MultipliedOperator) = LazySum([op1, op2])

Base.repeat(x::LazySum, args...) = LazySum(repeat.(x, args...))

function Base.getproperty(sumops::LazySum{<:Window}, sym::Symbol)
    if sym === :left || sym === :middle || sym === :right
        #extract the left/right parts
        return map(x -> getproperty(x, sym), sumops)
    else
        return getfield(sumops, sym)
    end
end
