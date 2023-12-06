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

"""
    Structure representing a time-independent operator that will be multiplied with a constant coefficient. Consists of
        - An operator (MPO, Hamiltonian, ...)
        - A number f that gets multiplied with the operator
"""
const UntimedOperator{O} = MultipliedOperator{O,<:Union{Real,One}}

#constructors for (un)TimedOperator
TimedOperator(x::O, f::F) where {F<:Function,O} = MultipliedOperator(x, f)
UntimedOperator(x::O, c::C) where {C<:Union{Real,One},O} = MultipliedOperator(x, c)

TimedOperator(x) = TimedOperator(x,t->One())
UntimedOperator(x) = UntimedOperator(x,One())

# For internal use only
unsafe_eval(x::UntimedOperator) = x.f * x.op
unsafe_eval(x::UntimedOperator, ::Number) = unsafe_eval(x)
unsafe_eval(x::TimedOperator, t::Number) = UntimedOperator(x.op,x.f(t))

# For users
(x::UntimedOperator)()  = unsafe_eval(x)
(x::TimedOperator)(t::Number)  = unsafe_eval(x,t)

# what to do when we multiply by a scalar
Base.:*(op::UntimedOperator, b::Number) = UntimedOperator(op.op, b * op.f)
Base.:*(op::TimedOperator, b::Number)   = TimedOperator(op.op, t -> b * op.f(t))
Base.:*(b::Number, op::MultipliedOperator) = op * b

Base.:*(op::TimedOperator, g::Function) = TimedOperator(op.op, t -> g(t) * op.f(t)) #slightly dangerous
Base.:*(g::Function, op::TimedOperator) = op * g

# don't know a better place to put this
# environment for MultipliedOperator
function environments(st, x::MultipliedOperator, args...; kwargs...)
    return environments(st, x.op, args...; kwargs...)
end