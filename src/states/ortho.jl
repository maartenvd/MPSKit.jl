"""
    UniformGauging <: Algorithm

Parameters for the uniform gauging algorithm.

## Fields
- `tol::Float64`: Tolerance for convergence.
- `maxiter::Int`: Maximum number of iterations.
- `verbosity::Int`: Verbosity level.
- `alg_leftorth`: Left-orthogonalization algorithm.
- `alg_rightorth`: Right-orthogonalization algorithm.
- `alg_eigsolve`: Eigensolver algorithm.
- `eig_miniter::Int`: Minimum number of iterations before eigensolver is used.
- `order::Symbol=:LR`: Order of the gauge algorithm.
"""
struct UniformGauging{side} <: Algorithm
    tol::Float64
    maxiter::Int
    verbosity::Int

    alg_orth::Any
    alg_eigsolve::Any
    eig_miniter::Int
end

function UniformGauging(;
                        tol=Defaults.tolgauge,
                        maxiter=Defaults.maxiter,
                        verbosity=VERBOSE_WARN,
                        alg_leftorth=QRpos(),
                        alg_rightorth=LQpos(),
                        alg_eigsolve=default_gauge_alg_eigsolve(tol, maxiter),
                        eig_miniter=10,
                        order::Symbol=:LR)
    alg_left = UniformGauging{:L}(tol, maxiter, verbosity, alg_leftorth, alg_eigsolve,
                                  eig_miniter)
    alg_right = UniformGauging{:R}(tol, maxiter, verbosity, alg_rightorth, alg_eigsolve,
                                   eig_miniter)
    if order === :LR
        return (alg_left, alg_right)
    elseif order === :RL
        return (alg_right, alg_left)
    else
        throw(ArgumentError("invalid order: $order"))
    end
end

const UG{side1,side2} = Tuple{UniformGauging{side1},UniformGauging{side2}}

# Interface
# ---------

function gaugefix!(ψ::InfiniteMPS; kwargs...)
    alg = UniformGauging(; kwargs...)
    return gaugefix!(ψ, alg)
end

function gaugefix!(ψ::InfiniteMPS, alg::UniformGauging)
    solve!(ψ, alg)
    return ψ
end

# expert mode
function gaugefix!(ψ::InfiniteMPS,
                   (alg₁, alg₂)::UG{side1,side2}) where {side1,side2}
    if side1 === :L
        copy!.(ψ.AR, ψ.AC)
    else
        copy!.(ψ.AL, ψ.AC)
    end

    solve!(ψ.AL, ψ.AR, ψ.CR, alg₁)
    solve!(ψ.AL, ψ.AR, ψ.CR, alg₂)
    mul!.(ψ.AC, ψ.AL, ψ.CR)
    return ψ
end

# Implementation
# --------------

function initialize!(AL::PeriodicVector{A},
                     AR::PeriodicVector{A},
                     CR::PeriodicVector{B},
                     alg::UniformGauging{side}) where {side,S,A<:GenericMPSTensor{S},
                                                       B<:MPSBondTensor{S}}
    if side === :L
        log = IterLog("UniformGauging{:L}")
        A_tail = _transpose_tail.(AR)
        CA_tail = similar.(A_tail)
        workspace = (; A_tail, CA_tail)
    else
        log = IterLog("UniformGauging{:R}")
        AC_tail = _similar_tail.(AL)
        workspace = (; AC_tail)
    end

    state = (; AL, AR, CR, workspace, ϵ=Inf, iter=0, log)
    loginit!(log, Inf)

    return IterativeSolver(alg, state)
end

function isfinished(alg::IterativeSolver{<:UniformGauging})
    return alg.state.ϵ < alg.alg.tol
end

function iscancelled(alg::IterativeSolver{<:UniformGauging})
    return alg.state.iter ≥ alg.alg.maxiter
end

function iterate!(it::IterativeSolver{UniformGauging{:L}})
    (; AL, AR, CR, workspace, iter, ϵ, log) = it.state
    alg = it.alg
    if iter ≥ alg.eig_miniter
        # attempt to replicate previous code: tol = max(ϵ², tol / 10)
        alg_eigsolve = updatetol(alg.alg_eigsolve, 1, ϵ^2)
        _, vecs = eigsolve(flip(TransferMatrix(AR, AL)), CR[end], 1, :LM, alg_eigsolve)
        _, CR[end] = leftorth!(vecs[1]; alg=alg.alg_orth)
    end

    C₀ = CR[end]
    for i in 1:length(AL)
        mul!(workspace.CA_tail[i], CR[i - 1], workspace.A_tail[i])
        _repartition!(AL[i], workspace.CA_tail[i])
        AL[i], CR[i] = leftorth!(AL[i]; alg=alg.alg_orth)
    end
    normalize!(CR[end])

    ϵ = oftype(ϵ, norm(C₀ - CR[end]))
    iter += 1

    it.state = (; AL, AR, CR, workspace, ϵ, iter, log)

    return AL, AR, CR
end

function iterate!(it::IterativeSolver{UniformGauging{:R}})
    (; AL, AR, CR, workspace, iter, ϵ, log) = it.state
    alg = it.alg
    # eigsolve step
    if iter ≥ alg.eig_miniter
        # attempt to replicate previous code: tol = max(ϵ², tol / 10)
        alg_eigsolve = updatetol(alg.alg_eigsolve, 1, ϵ^2)
        _, vecs = eigsolve(TransferMatrix(AL, AR), CR[end], 1, :LM, alg_eigsolve)
        CR[end], _ = rightorth!(vecs[1]; alg=alg.alg_orth)
    end

    # rightorth step
    C₀ = CR[end]
    for i in length(AR):-1:1
        mul!(AR[i], AL[i], CR[i]) # use AR as temporary storage for A-C
        tmp = _repartition!(workspace.AC_tail[i], AR[i])
        CR[i - 1], tmp = rightorth!(tmp; alg=alg.alg_orth)
        _repartition!(AR[i], tmp)
    end
    normalize!(CR[end])

    ϵ = oftype(ϵ, norm(C₀ - CR[end]))
    iter += 1

    it.state = (; AL, AR, CR, workspace, ϵ, iter, log)

    return AL, AR, CR
end

function finalize!(it::IterativeSolver{<:UniformGauging})
    return it.state.AL, it.state.AR, it.state.CR
end

function logiter(it::IterativeSolver{<:UniformGauging})
    @infov 3 logiter!(it.state.log, it.state.iter, it.state.ϵ)
end

function logfinish(it::IterativeSolver{<:UniformGauging})
    @infov 2 logfinish!(it.state.log, it.state.iter, it.state.ϵ)
end

function logcancel(it::IterativeSolver{<:UniformGauging})
    @warnv 1 logcancel!(it.state.log, it.state.iter, it.state.ϵ)
end

# ------------------------------------------------------------------------------------------
# Re-gauging AC and C
# ------------------------------------------------------------------------------------------

function regauge!(AC::PeriodicVector{<:GenericMPSTensor},
                  CR::PeriodicVector{<:MPSBondTensor}, alg::QRpos=QRpos())
    for i in 1:length(AC)
        # find AL that best fits these new AC and CR
        QAc, _ = leftorth!(AC[i]; alg)
        Qc, _ = leftorth!(CR[i]; alg)
        mul!(AC[i], QAc, Qc')
    end
    return AC
end

# Utility
# -------

function default_gauge_alg_eigsolve(tol, maxiter)
    eigalg = Arnoldi(; krylovdim=30, maxiter)
    tol_min = tol / 10
    tol_max = Inf
    tol_factor = 1
    return DynamicTol(eigalg, tol_min, tol_max, tol_factor)
end

# function regauge!(AC::PeriodicVector{<:GenericMPSTensor},
#                   CR::PeriodicVector{<:MPSBondTensor}, alg::LQpos)
#     for i in 1:length(AC)
#         # find AR that best fits these new AC and CR
#         AC_tail = _transpose_tail(AC[i])
#         _, QAc = rightorth!(AC_tail; alg)
#         _, Qc = rightorth!(CR[i - 1]; alg)
#         mul!(AC_tail, Qc', QAc)
#         _repartition!(AC[i], AC_tail)
#     end
#     return AC
# end

# first iteration:
#   needs to set up correct tensors with correct spaces
#   - check all A full-rank
#   - make all C square
#   - allocate temporary storage

# @assert domain(it.CR[end]) == codomain(it.CR[end]) "C₀ must be square: $(space(it.CR[end]))"

#     # make everything full rank in order to keep spaces fixed afterwards
#     # this should result in smallest injective MPS of same state
#     while true
#         flag = false
#         for i in 1:length(it.A)
#             # Vₗ ⊗ P ≿ Vᵣ
#             if !(codomain(it.A[i]) ≾ domain(it.A[i]))
#                 # shrink right virtual space
#                 it.A[i], R = leftorth!(it.A[i]; alg=it.leftorthalg)
#                 it.A[i + 1] = _transpose_front(R * _transpose_tail(it.A[i + 1]))
#                 flag = true
#             end

#             # Vₗ ≾ P' ⊗ Vᵣ
#             if !(_firstspace(it.A[i]) ≾
#                  ⊗(space.(Ref(it.A[i + 1]), 2:numind(it.A[i + 1]))...))
#                 # shrink left virtual space
#                 L, Q = rightorth!(it.A[i]; alg=it.rightorthalg)
#                 it.A[i - 1] = it.A[i - 1] * L
#                 it.A[i] = _transpose_front(Q)
#                 flag = true
#             end
#         end
#         flag || break
#     end
