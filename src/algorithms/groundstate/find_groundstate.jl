"""
    find_groundstate(ψ, H, [environments]; kwargs...)
    find_groundstate(ψ, H, algorithm, environments)

Compute the groundstate for Hamiltonian `H` with initial guess `ψ`. If not specified, an
optimization algorithm will be attempted based on the supplied keywords.

## Arguments
- `ψ::AbstractMPS`: initial guess
- `H::AbstractMPO`: operator for which to find the groundstate
- `[environments]`: MPS environment manager
- `algorithm`: optimization algorithm

## Keywords
- `tol::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `verbosity::Int`: display progress information
"""
function find_groundstate(ψ::AbstractMPS, H,
                          envs::Union{Cache,MultipleEnvironments}=environments(ψ, H);
                          tol=Defaults.tol, maxiter=Defaults.maxiter,
                          verbosity=Defaults.verbosity, trscheme=nothing)
    if isa(ψ, InfiniteMPS)
        alg = VUMPS(; tol=max(1e-4, tol), verbosity, maxiter)
        if tol < 1e-4
            alg = alg &
                  GradientGrassmann(; tol=tol, maxiter, verbosity)
        end
        if !isnothing(trscheme)
            alg = IDMRG2(; tol=min(1e-2, 100tol), verbosity,
                         trscheme) & alg
        end
    elseif isa(ψ, AbstractFiniteMPS)
        alg = DMRG(; tol, maxiter, verbosity)
        if !isnothing(trscheme)
            alg = DMRG2(; tol=min(1e-2, 100tol), verbosity, trscheme) & alg
        end
    else
        throw(ArgumentError("Unknown input state type"))
    end
    if isa(ψ, WindowMPS)
        alg_infin = VUMPS(; tol=tol, verbosity=verbosity, maxiter=maxiter)
        alg = Window(alg_infin, alg, alg_infin)
    end
    return find_groundstate(ψ, H, alg, envs)
end

function find_groundstate!(state::WindowMPS{A,B,VL,VR}, H::Union{Window,LazySum{<:Window}},
                           alg::Window, envs=environments(state, H)) where {A,B,VL,VR}
    # first find infinite groundstates
    if VL === WINDOW_VARIABLE
        (gs_left, _) = find_groundstate(state.left, H.left, alg.left, envs.left)
        state = WindowMPS(gs_left, state.middle, state.right)
    end
    if VR === WINDOW_VARIABLE
        (gs_right, _) = find_groundstate(state.right, H.right, alg.right, envs.right)
        state = WindowMPS(state.left, state.middle, gs_right)
    end
    # then find finite groundstate
    state, _, delta = find_groundstate(state, H.middle, alg.middle,
                                       finenv(envs, state))
    return state, envs, delta
end

function find_groundstate(ψ::WindowMPS, H::Union{Window,LazySum{<:Window}}, alg::Window,
                          envs...)
    return find_groundstate!(copy(ψ), H, alg, envs...)
end
