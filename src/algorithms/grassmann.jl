# We separate some of the internals needed for implementing GradientGrassmann into a
# submodule, to keep the MPSKit module namespace cleaner.
"""
A module for functions related to treating an InfiniteMPS in left-canonical form as a bunch
of points on Grassmann manifolds, and performing things like retractions and transports
on these Grassmann manifolds.

The module exports nothing, and all references to it should be qualified, e.g.
`GrassmannMPS.fg`.
"""
module GrassmannMPS

using ..MPSKit
using TensorKit
import TensorKitManifolds.Grassmann

function TensorKit.rmul!(a::Grassmann.GrassmannTangent, b::AbstractTensorMap)
    #rmul!(a.Z,b);
    Base.setfield!(a, :Z, a.Z * b)
    Base.setfield!(a, :U, nothing)
    Base.setfield!(a, :S, nothing)
    Base.setfield!(a, :V, nothing)
    return a
end

# preconditioned gradient
struct PrecGrad{A,B}
    Pg::A
    g::A
    rho::B
end

function PrecGrad(v::Grassmann.GrassmannTangent)
    return PrecGrad(v, v, isometry(storagetype(v.Z), domain(v.Z), domain(v.Z)))
end
PrecGrad(v::Grassmann.GrassmannTangent, rho) = PrecGrad(rmul!(copy(v), inv(rho)), v, rho)
Grassmann.base(g::PrecGrad) = Grassmann.base(g.Pg)

function inner(g1::PrecGrad, g2::PrecGrad, rho=one(g1.rho))
    Grassmann.base(g1) == Grassmann.base(g2) || throw(ArgumentError("incompatible base"))
    if g1.rho == rho
        dot(g1.g.Z, g2.Pg.Z)
    elseif g2.rho == rho
        dot(g1.Pg.Z, g2.g.Z)
    else
        dot(g1.Pg.Z, g2.Pg.Z * rho)
    end
end

Base.:*(g::PrecGrad, alpha::Number) = PrecGrad(g.Pg * alpha, g.g * alpha, g.rho)
function Base.:+(a::PrecGrad, b::PrecGrad)
    if a.rho == b.rho
        PrecGrad(a.Pg + b.Pg, a.g + b.g, a.rho)
    else
        PrecGrad(a.Pg + b.Pg)
    end
end

struct ManifoldPoint{T,E,G,C}
    state::T # the state at that point
    envs::E # the environments
    g::G # the MPS gradient, which is not equivalent with the grassmann gradient!
    Rhoreg::C # the regularized density matrices
end

function ManifoldPoint(state::Union{InfiniteMPS,FiniteMPS}, envs)
    al_d = similar(state.AL)
    for i in 1:length(state)
        al_d[i] = MPSKit.∂∂AC(i, state, envs.opp, envs) * state.AC[i]
    end

    g = Grassmann.project.(al_d, state.AL)

    Rhoreg = Vector{eltype(state.CR)}(undef, length(state))
    for i in 1:length(state)
        Rhoreg[i] = regularize(state.CR[i], norm(g[i]) / 10)
    end

    return ManifoldPoint(state, envs, g, Rhoreg)
end

function ManifoldPoint(state::MPSMultiline, envs)
    ac_d = [MPSKit.∂∂AC(v, state, envs.opp, envs) * state.AC[v]
            for v in CartesianIndices(state.AC)]
    g = [Grassmann.project(d, a) for (d, a) in zip(ac_d, state.AL)]

    f = expectation_value(state, envs)
    fi = imag.(f)
    fr = real.(f)

    sum(fi) > MPSKit.Defaults.tol && @warn "mpo is not hermitian $fi"

    g = -2 * g ./ abs.(fr)

    Rhoreg = similar(state.CR)
    for (i, cg) in enumerate(g)
        Rhoreg[i] = regularize(state.CR[i], norm(cg) / 10)
    end

    return ManifoldPoint(state, envs, g, Rhoreg)
end

"""
Compute the expectation value, and its gradient with respect to the tensors in the unit
cell as tangent vectors on Grassmann manifolds.
"""
function fg(x::ManifoldPoint{T}) where {T<:Union{InfiniteMPS,FiniteMPS}}
    # the gradient I want to return is the preconditioned gradient!
    g_prec = Vector{PrecGrad{eltype(x.g),eltype(x.Rhoreg)}}(undef, length(x.g))

    for i in 1:length(x.state)
        g_prec[i] = PrecGrad(rmul!(copy(x.g[i]), x.state.CR[i]'), x.Rhoreg[i])
    end

    f = real(sum(expectation_value(x.state, x.envs)))

    return f, g_prec
end
function fg(x::ManifoldPoint{<:MPSMultiline})
    # the gradient I want to return is the preconditioned gradient!
    g_prec = map(enumerate(x.g)) do (i, cg)
        return PrecGrad(rmul!(copy(cg), x.state.CR[i]'), x.Rhoreg[i])
    end

    f = expectation_value(x.state, x.envs)
    fi = imag.(f)
    fr = real.(f)

    return -log(sum(fr)^2), g_prec[:]
end

"""
Retract a left-canonical MPSMultiline along Grassmann tangent `g` by distance `alpha`.
"""
function retract(x::ManifoldPoint{<:MPSMultiline}, tg, alpha)
    g = reshape(tg, size(x.state))

    nal = similar(x.state.AL)
    h = similar(g)
    for (i, cg) in enumerate(tg)
        (nal[i], th) = Grassmann.retract(x.state.AL[i], cg.Pg, alpha)
        h[i] = PrecGrad(th)
    end

    nstate = MPSKit.MPSMultiline(nal, x.state.CR[:, end])
    newpoint = ManifoldPoint(nstate, x.envs)

    return newpoint, h[:]
end

"""
Retract a left-canonical infinite MPS along Grassmann tangent `g` by distance `alpha`.
"""
function retract(x::ManifoldPoint{<:InfiniteMPS}, g, alpha)
    ψ = x.state
    envs = x.envs
    ψ′ = copy(ψ)
    h = similar(g) # The tangent at the end-point
    for i in eachindex(g)
        ψ′.AC[i], th = Grassmann.retract(ψ.AL[i], g[i].Pg, alpha)
        h[i] = PrecGrad(th)
    end

    # important to do it like this because Grassmann retract checks if base point is the same
    # and `uniform_gauge()` creates new AL (I think?)
    gaugefix!(ψ′; order=:LR) # needs to be this order for efficiency

    # AR, CR = uniform_rightgauge(AL, ψ.CR[end])
    # ψ′ = InfiniteMPS(AL, AR, CR)

    newpoint = ManifoldPoint(ψ′, envs)

    return newpoint, h
end

"""
Retract a left-canonical finite MPS along Grassmann tangent `g` by distance `alpha`.
"""
function retract(x::ManifoldPoint{<:FiniteMPS}, g, alpha)
    state = x.state
    envs = x.envs

    y = copy(state)  # The end-point
    h = similar(g)  # The tangent at the end-point
    for i in 1:length(g)
        (yal, th) = Grassmann.retract(state.AL[i], g[i].Pg, alpha)
        h[i] = PrecGrad(th)
        y.AC[i] = (yal, state.CR[i])
    end
    normalize!(y)

    n_y = ManifoldPoint(y, envs)

    return n_y, h
end

"""
Transport a tangent vector `h` along the retraction from `x` in direction `g` by distance
`alpha`. `xp` is the end-point of the retraction.
"""
function transport!(h, x, g, alpha, xp)
    for i in 1:length(h)
        h[i] = PrecGrad(Grassmann.transport!(h[i].Pg, x.state.AL[i], g[i].Pg, alpha,
                                             xp.state.AL[i]))
    end
    return h
end

"""
Euclidean inner product between two Grassmann tangents of an infinite MPS.
"""
function inner(x, g1, g2)
    return 2 * real(sum(((a, b, c),) -> inner(b, c, a), zip(x.Rhoreg, g1, g2)))
end

"""
Scale a tangent vector by scalar `alpha`.
"""
scale!(g, alpha) = g .* alpha

"""
Add two tangents vectors, scaling the latter by `alpha`.
"""
add!(g1, g2, alpha) = g1 + g2 .* alpha

"""
Take the L2 Tikhonov regularised of a matrix `m`.

The regularisation parameter is the larger of `delta` (the optional argument that defaults
to zero) and square root of machine epsilon.
"""
function regularize(m, delta=zero(scalartype(m)))
    U, S, V = tsvd(m)

    #Sreg = real(S*sqrt(one(S) + delta^2*one(S)*norm(S,Inf)^2/S^2));#
    Sreg = S^2 + (norm(S, Inf) * delta)^2 * one(S)

    Mreg = U * Sreg * U'

    return Mreg
end

end  # module GrassmannMPS
