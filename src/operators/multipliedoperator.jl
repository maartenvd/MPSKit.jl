"""
    Structure representing a multiplied operator. Consists of
        - An operator op (MPO, Hamiltonian, ...)
        - An object f that gets multiplied with the operator (Number, function, ...) 
"""
struct MultipliedOperator{O,F}
    op::O
    f::F
end

"""
    Structure representing a time-dependent operator. Consists of
        - An operator op (MPO, Hamiltonian, ...)
        - An function f that gives the time-dependence according to op(t) = f(t)*op
"""
const TimedOperator{O} = MultipliedOperator{O,<:Function}
TimedOperator(x) = MultipliedOperator(x, t -> One())

"""
    Structure representing a time-independent operator that will be multiplied with a constant coefficient. Consists of
        - An operator (MPO, Hamiltonian, ...)
        - A number f that gets multiplied with the operator
"""
const UntimedOperator{O} = MultipliedOperator{O,<:Number}
UntimedOperator(x) = MultipliedOperator(x, One())

# Holy traits
TimeDependence(::TimedOperator) = TimeDependent()

# For internal use only
_eval_at(x::UntimedOperator) = x.f * x.op
_eval_at(x::UntimedOperator, ::Number) = x
_eval_at(x::TimedOperator, t::Number) = MultipliedOperator(x.op, x.f(t))

# For users
(x::UntimedOperator)() = _eval_at(x)
(x::TimedOperator)(t::Number) = _eval_at(x, t)

# what to do when we multiply by a scalar
Base.:*(op::UntimedOperator, b::Number) = MultipliedOperator(op.op, b * op.f)
Base.:*(op::TimedOperator, b::Number) = MultipliedOperator(op.op, t -> b * op.f(t))
Base.:*(b, op::MultipliedOperator) = op * b

# slightly dangerous
Base.:*(op::TimedOperator, g::Function) = MultipliedOperator(op.op, t -> g(t) * op.f(t))
Base.:*(op::UntimedOperator, g::Function) = MultipliedOperator(op.op, t -> g(t) * op.f)

# don't know a better place to put this
# environment for MultipliedOperator
function environments(st, x::MultipliedOperator, args...; kwargs...)
    return environments(st, x.op, args...; kwargs...)
end

function MultipliedOperator(x::Window, f)
    return Window(MultipliedOperator(x.left, f), MultipliedOperator(x.middle, f),
                  MultipliedOperator(x.right, f))
end