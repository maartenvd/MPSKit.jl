"""
    abstract type DDMRG_Flavour end

Abstract supertype for the different flavours of dynamical DMRG.
"""
abstract type DDMRG_Flavour end

"""
    struct DynamicalDMRG{F,S} <: Algorithm end

A dynamical DMRG method for calculating dynamical properties and excited states, based on a
variational principle for dynamical correlation functions.

The algorithm is described in detail in https://arxiv.org/pdf/cond-mat/0203500.pdf.

# Fields
- `flavour::F = NaiveInvert` : The flavour of the algorithm to use. Currently only `NaiveInvert` and `Jeckelmann` are implemented.
- `solver::S = Defaults.linearsolver` : The linear solver to use for the linear systems.
- `tol::Float64 = Defaults.tol * 10` : The stopping criterium.
- `maxiter::Int = Defaults.maxiter` : The maximum number of iterations.
- `verbosity::Int = Defaults.verbosity` : Whether to print information about the progress of the algorithm.
"""
@kwdef struct DynamicalDMRG{F<:DDMRG_Flavour,S} <: Algorithm
    flavour::F = NaiveInvert
    solver::S = Defaults.linearsolver
    tol::Float64 = Defaults.tol * 10
    maxiter::Int = Defaults.maxiter
    verbosity::Int = Defaults.verbosity
end

"""
    propagator(ψ₀::AbstractFiniteMPS, z::Number, H::MPOHamiltonian, alg::DynamicalDMRG; init=copy(ψ₀))

Calculate the propagator ``\\frac{1}{E₀ + z - H}|ψ₀>`` using the dynamical DMRG
algorithm.
"""
function propagator end

"""
    struct NaiveInvert <: DDMRG_Flavour end

An alternative approach to the dynamical DMRG algorithm, without quadratic terms but with a
less controlled approximation.

This algorithm essentially minimizes ``<ψ|(H - E)|ψ> - <ψ|ψ₀> - <ψ₀|ψ>``, which is
equivalent to the original approach if ``|ψ₀> = (H - E)|ψ>``.
"""
struct NaiveInvert <: DDMRG_Flavour end

function propagator(A::AbstractFiniteMPS, z::Number, H::MPOHamiltonian,
                    alg::DynamicalDMRG{NaiveInvert}; init=copy(A))
    h_envs = environments(init, H) # environments for h
    mixedenvs = environments(init, A) # environments for <init | A>

    ϵ = 2 * alg.tol
    log = IterLog("DDMRG")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            ϵ = 0.0

            for i in [1:(length(A) - 1); length(A):-1:2]
                tos = ac_proj(i, init, mixedenvs)

                H_AC = ∂∂AC(i, init, H, h_envs)
                AC = init.AC[i]
                AC′, convhist = linsolve(H_AC, -tos, AC, alg.solver, -z, one(z))

                ϵ = max(ϵ, norm(AC′ - AC))
                init.AC[i] = AC′

                convhist.converged == 0 &&
                    @warn "propagator ($i) failed to converge: normres = $(convhist.normres)"
            end

            if ϵ <= alg.tol
                @infov 2 logfinish!(log, iter, ϵ)
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ)
            else
                @infov 3 logiter!(log, iter, ϵ)
            end
        end
    end

    return dot(A, init), init
end

"""
    struct Jeckelmann <: DDMRG_Flavour end

The original flavour of dynamical DMRG, as described in
https://arxiv.org/pdf/cond-mat/0203500.pdf. The algorithm minimizes
``||(H - E)|ψ₀> - |ψ>||``, thus containing quadratic terms in ``H - E``.
"""
struct Jeckelmann <: DDMRG_Flavour end

function propagator(A::AbstractFiniteMPS, z, H::MPOHamiltonian,
                    alg::DynamicalDMRG{Jeckelmann}; init=copy(A))
    ω = real(z)
    η = imag(z)

    envs1 = environments(init, H) # environments for h
    H2, envs2 = squaredenvs(init, H, envs1) # environments for h^2
    mixedenvs = environments(init, A) # environments for <init | A>

    ϵ = 2 * alg.tol
    log = IterLog("DDMRG")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            ϵ = 0.0

            for i in [1:(length(A) - 1); length(A):-1:2]
                tos = ac_proj(i, init, mixedenvs)
                H1_AC = ∂∂AC(i, init, H, envs1)
                H2_AC = ∂∂AC(i, init, H2, envs2)
                H_AC = LinearCombination((H1_AC, H2_AC), (-2 * ω, 1))
                AC′, convhist = linsolve(H_AC, -η * tos, init.AC[i], alg.solver, abs2(z), 1)

                ϵ = max(ϵ, norm(AC′ - init.AC[i]))
                init.AC[i] = AC′

                convhist.converged == 0 &&
                    @warn "propagator ($i) failed to converge: normres $(convhist.normres)"
            end

            if ϵ <= alg.tol
                @infov 2 logfinish!(log, iter, ϵ)
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ)
            else
                @infov 3 logiter!(log, iter, ϵ)
            end
        end
    end

    a = dot(ac_proj(1, init, mixedenvs), init.AC[1])
    cb = leftenv(envs1, 1, A) * TransferMatrix(init.AL, H[1:length(A.AL)], A.AL)
    b = zero(a)
    for i in 1:length(cb)
        b += @plansor cb[i][1 2; 3] * init.CR[end][3; 4] *
                      rightenv(envs1, length(A), A)[i][4 2; 5] * conj(A.CR[end][1; 5])
    end

    v = b / η - ω / η * a + 1im * a
    return v, init
end

function squaredenvs(Ψ, H, envs, H2, envs2)
    L = length(Ψ)
    # make sure the dependencies are incorrect, so data will be recalculated
    for i in 1:L
        invalidate!(envs2, i)
    end

    # impose the correct boundary conditions
    # (important for comoving mps, should do nothing for finite mps)
    indmap = LinearIndices((H.odim, H.odim))
    @sync begin
        Threads.@spawn begin
            nleft = leftenv(envs2, 1, Ψ)
            for i in 1:(H.odim), j in 1:(H.odim)
                nleft[indmap[i, j]] = _contract_leftenv²(leftenv(envs, 1, Ψ)[j],
                                                         leftenv(envs, 1, Ψ)[i])
            end
        end
        Threads.@spawn begin
            nright = rightenv(envs2, L, Ψ)
            for i in 1:(H.odim), j in 1:(H.odim)
                nright[indmap[i, j]] = _contract_rightenv²(rightenv(envs, L, Ψ)[j],
                                                           rightenv(envs, L, Ψ)[i])
            end
        end
    end
    return H2, envs2
end

function squaredenvs(Ψ::AbstractFiniteMPS, H::MPOHamiltonian,
                     envs::FinEnv=environments(Ψ, H))
    # to construct the squared caches we will first initialize environments
    # then make all data invalid so it will be recalculated
    # then initialize the correct caches at the edge
    nH = conj(H) * H
    return squaredenvs(Ψ, H, envs, nH, environments(Ψ, nH))
end

function squaredenvs(Ψ::WindowMPS, H::Window, envs::WindowEnv=environments(Ψ, H))
    nH = Window(conj(H.left) * H.left, conj(H.middle) * H.middle, conj(H.right) * H.right)
    return squaredenvs(Ψ, H.middle, envs, nH, environments(Ψ, nH))
end

function _contract_leftenv²(GL_top, GL_bot)
    V_mid = space(GL_bot, 2)' ⊗ space(GL_top, 2)
    F = isomorphism(storagetype(GL_top), fuse(V_mid)' ← V_mid)
    return @plansor GL[-1 -2; -3] := GL_top[1 3; -3] * conj(GL_bot[1 2; -1]) * F[-2; 2 3]
end

function _contract_rightenv²(GR_top, GR_bot)
    V_mid = space(GR_top, 2) ⊗ space(GR_bot, 2)'
    F = isomorphism(storagetype(GR_top), fuse(V_mid) ← V_mid)
    return @plansor GR[-1 -2; -3] := GR_top[-1 2; 1] * conj(GR_bot[-3 3; 1]) * F[-2; 2 3]
end
