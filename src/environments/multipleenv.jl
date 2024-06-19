struct MultipleEnvironments{O,C} <: Cache
    opp::O
    envs::Vector{C}
end

Base.size(x::MultipleEnvironments) = size(x.envs)
Base.getindex(x::MultipleEnvironments, i) = x.envs[i]
Base.length(x::MultipleEnvironments) = prod(size(x))

Base.iterate(x::MultipleEnvironments) = iterate(x.envs)
Base.iterate(x::MultipleEnvironments, i) = iterate(x.envs, i)

# we need constructor, agnostic of particular MPS
function environments(st, H::LazySum)
    return MultipleEnvironments(H, map(op -> environments(st, op), H.ops))
end

function environments(st::Union{InfiniteMPS,MPSMultiline}, H::LazySum;
                      solver=Defaults.linearsolver)
    if !(solver isa Vector)
        solver = repeat([solver], length(H))
    end
    return MultipleEnvironments(H,
                                map((op, solv) -> environments(st, op; solver=solv),
                                    H.ops, solver))
end

#broadcast vs map?
# function environments(state, H::LinearCombination)
#     return MultipleEnvironments(H, broadcast(o -> environments(state, o), H.opps))
# end;

#===========================================================================================
Utility
===========================================================================================#
function Base.getproperty(ca::MultipleEnvironments{<:LazySum,<:WindowEnv}, sym::Symbol)
    if sym === :left || sym === :middle || sym === :right
        #extract the left/right parts
        return MultipleEnvironments(getproperty(ca.opp, sym),
                                    map(x -> getproperty(x, sym), ca))
    else
        return getfield(ca, sym)
    end
end

function Base.getproperty(envs::MultipleEnvironments, prop::Symbol)
    if prop === :solver
        return map(env -> env.solver, envs)
    else
        return getfield(envs, prop)
    end
end

function finenv(ca::MultipleEnvironments{<:LazySum,<:WindowEnv}, ψ::WindowMPS)
    return MultipleEnvironments(ca.opp.middle, map(x -> finenv(x, ψ), ca))
end

# we need to define how to recalculate
"""
    Recalculate in-place each sub-env in MultipleEnvironments
"""
function recalculate!(env::MultipleEnvironments, args...; kwargs...)
    for subenv in env.envs
        recalculate!(subenv, args...; kwargs...)
    end
    return env
end
