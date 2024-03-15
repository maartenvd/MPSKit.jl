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
struct UniformGauging <: Algorithm
    tol::Float64
    maxiter::Int
    verbosity::Int

    alg_leftorth::Any
    alg_rightorth::Any
    alg_eigsolve::Any
    eig_miniter::Int

    order::Symbol
end

function UniformGauging(;
                        tol=Defaults.tolgauge,
                        maxiter=Defaults.maxiter,
                        verbosity=VERBOSE_WARN,
                        alg_leftorth=QRpos(),
                        alg_rightorth=TensorKit.LQpos(),
                        alg_eigsolve=default_gauge_alg_eigsolve(tol, maxiter),
                        eig_miniter=10,
                        order::Symbol=:LR)
    return UniformGauging(tol, maxiter, verbosity, alg_leftorth, alg_rightorth,
                          alg_eigsolve,
                          eig_miniter, order)
end

"""
    uniform_gauge(A, [C₀]; kwargs...) -> AL, AR, CR

Brings an infinite MPS, characterized by the tensors `A`, into the center gauge. Optionally,
a gauge tensor `C₀` can be provided to seed the algorithm. The algorithm parameters are set
via keyword arguments.

## Keyword Arguments
- `tol::Real=Defaults.tolgauge`: Tolerance for the gauge convergence.
- `maxiter::Int=Defaults.maxiter`: Maximum number of iterations.
- `verbosity::Int=VERBOSE_WARN`: Verbosity level.
- `order::Symbol=:LR`: Order of the gauge algorithm. Can be `:LR` or `:RL`.
- `alg_leftorth=QRpos()`: Left-orthogonalization algorithm.
- `alg_rightorth=TensorKit.LQpos()`: Right-orthogonalization algorithm.
- `alg_eigsolve=default_gauge_alg_eigsolve(tol, maxiter)`:
    Eigensolver algorithm for the gauge tensor.
- `eig_miniter::Int=10`: Minimum number of iterations before eigensolver is used.
"""
function uniform_gauge(A::AbstractVector{<:GenericMPSTensor}, C₀=gauge_init(A);
                       kwargs...)
    alg = UniformGauging(; kwargs...)
    return uniform_gauge(A, C₀, alg)
end
function uniform_gauge(A::AbstractVector{<:GenericMPSTensor}, C₀::MPSBondTensor,
                       alg::UniformGauging)
    if alg.order === :LR
        AL, CR = uniform_leftgauge(A, C₀, alg)
        AR, CR = uniform_rightgauge(AL, CR[end], alg)
    elseif alg.order === :RL
        AR, CR = uniform_rightgauge(A, C₀, alg)
        AL, CR = uniform_leftgauge(AR, CR[end], alg)
    else
        throw(ArgumentError("invalid order: $order"))
    end
    return AL, AR, CR
end

"""
    uniform_leftgauge!(AL, CR, A, alg::UniformGauging)

Solves `AL[i] * CR[i] = CR[i] * A[i+1]`.
"""
function uniform_leftgauge(A::PeriodicVector{TA}, C₀::TB,
                           alg::UniformGauging=UniformGauging()) where {S,
                                                                        TA<:GenericMPSTensor{S},
                                                                        TB<:MPSBondTensor{S}}
    iterable = LeftGaugeIterable(A, C₀; alg.tol, alg.maxiter, alg.verbosity,
                                 alg.alg_leftorth, alg.alg_eigsolve, alg.eig_miniter)
    for _ in iterable
    end
    return iterable.AL, iterable.CR
end

"""
    uniform_rightgauge!(AR, CR, A, alg::UniformGauging)

Solves C[i-1] * AR[i] = A[i] * C[i].
"""
function uniform_rightgauge(A::PeriodicVector{TA}, C₀::TB,
                            alg::UniformGauging=UniformGauging()) where {S,
                                                                         TA<:GenericMPSTensor{S},
                                                                         TB<:MPSBondTensor{S}}
    iterable = RightGaugeIterable(A, C₀; alg.tol, alg.maxiter, alg.verbosity,
                                  alg.alg_rightorth, alg.alg_eigsolve, alg.eig_miniter)
    for _ in iterable
    end
    return iterable.AR, iterable.CR
end

function default_gauge_alg_eigsolve(tol, maxiter)
    eigalg = Arnoldi(; krylovdim=30, maxiter)
    tol_min = tol / 10
    tol_max = Inf
    tol_factor = 1
    return ThrottledTol(eigalg, tol_min, tol_max, tol_factor)
end

function gauge_init(A::AbstractVector{<:GenericMPSTensor})
    C = isomorphism(storagetype(A[1]), _firstspace(A[1]),
                    _firstspace(A[1]))
    return C
end

# ------------------------------------------------------------------------------------------
# Left Gauge
# ------------------------------------------------------------------------------------------

mutable struct LeftGaugeIterable{TA<:GenericMPSTensor,TAᵀ<:AbstractTensorMap,
                                 TB<:MPSBondTensor,T<:Number,Alg₁,Alg₂}
    # input MPS tensors
    A::PeriodicVector{TA}
    A_tail::PeriodicVector{TAᵀ}
    C::TB # gauge tensor left of unit cell

    # output tensors and workspace
    AL::PeriodicVector{TA}
    CA_tail::PeriodicVector{TAᵀ}
    CR::PeriodicVector{TB}
    δ::T

    # algorithm parameters
    tol::T
    maxiter::Int
    verbosity::Int

    alg_leftorth::Alg₁
    alg_eigsolve::Alg₂
    eig_miniter::Int

    function LeftGaugeIterable(A::PeriodicVector{TA}, C::TB,
                               AL::PeriodicVector{TA}=similar.(A),
                               CR::PeriodicVector{TB}=similar(A, TB),
                               A_tail::PeriodicVector{TAᵀ}=_transpose_tail.(A),
                               CA_tail::PeriodicVector{TAᵀ}=similar.(A_tail);
                               tol=Defaults.tolgauge,
                               δ=Inf,
                               maxiter=Defaults.maxiter,
                               verbosity=VERBOSE_WARN,
                               alg_leftorth::Alg₁=QRpos(),
                               alg_eigsolve::Alg₂=default_gauge_alg_eigsolve(tol, maxiter),
                               eig_miniter=10) where {TA,TAᵀ,TB,Alg₁,Alg₂}
        @assert all(isfullrank, A) "input tensors must be full rank.\n$(space.(A))"
        @assert domain(C) == codomain(C) "C must be square: $(space(C))"
        scalartype(A) === scalartype(C) ||
            throw(ArgumentError("A and C must have the same scalar type"))
        T = real(scalartype(A))
        return new{TA,TAᵀ,TB,T,Alg₁,Alg₂}(A, A_tail, C, AL, CA_tail, CR,
                                          T(δ), T(tol), maxiter, verbosity,
                                          alg_leftorth, alg_eigsolve, eig_miniter)
    end
end

function Base.iterate(it::LeftGaugeIterable, iteration::Int=0)
    # check for termination
    if it.δ ≤ it.tol
        it.verbosity ≥ VERBOSE_CONVERGENCE &&
            @info "leftgauge converged after $iteration iterations: δ = $(it.δ)"
        return nothing
    elseif iteration ≥ it.maxiter
        it.verbosity ≥ VERBOSE_WARN &&
            @warn "leftgauge not converged after $iteration iterations: δ = $(it.δ)"
        return nothing
    end

    # eigsolve step
    if iteration ≥ it.eig_miniter
        # attempt to replicate previous code: tol = max(δ², tol / 10)
        alg_eigsolve = updatetol(it.alg_eigsolve, 1, it.δ^2)
        _, vecs = eigsolve(flip(TransferMatrix(it.A, it.AL)), it.C, 1, :LM, alg_eigsolve)
        _, it.C = leftorth!(vecs[1]; alg=it.alg_leftorth)
    end

    # leftorth step
    it.CR[end] = it.C
    for i in 1:length(it.AL)
        mul!(it.CA_tail[i], it.CR[i - 1], it.A_tail[i])
        _repartition!(it.AL[i], it.CA_tail[i])
        it.AL[i], it.CR[i] = leftorth!(it.AL[i]; alg=it.alg_leftorth)
    end
    normalize!(it.CR[end])

    # check for convergence
    it.δ = norm(it.C - it.CR[end])
    it.C = it.CR[end]

    it.verbosity ≥ VERBOSE_ITER && @info "leftgauge iteration $iteration: δ = $(it.δ)"

    return it.C, iteration + 1
end

# ------------------------------------------------------------------------------------------
# Right Gauge
# ------------------------------------------------------------------------------------------

mutable struct RightGaugeIterable{TA<:GenericMPSTensor,TAᵀ<:AbstractTensorMap,
                                  TB<:MPSBondTensor,T<:Number,Alg₁,Alg₂}
    # input MPS tensors
    A::PeriodicVector{TA}
    C::TB # gauge tensor left of unit cell

    # output tensors and workspace
    AR::PeriodicVector{TA}
    AC_tail::PeriodicVector{TAᵀ}
    CR::PeriodicVector{TB}
    δ::T

    # algorithm parameters
    tol::T
    maxiter::Int
    verbosity::Int

    alg_rightorth::Alg₁
    alg_eigsolve::Alg₂
    eig_miniter::Int

    function RightGaugeIterable(A::PeriodicVector{TA}, C::TB,
                                AR::PeriodicVector{TA}=similar.(A),
                                CR::PeriodicVector{TB}=similar(A, TB),
                                AC_tail::PeriodicVector{TAᵀ}=_similar_tail.(A);
                                tol=Defaults.tolgauge,
                                δ=Inf,
                                maxiter=Defaults.maxiter,
                                verbosity=VERBOSE_WARN,
                                alg_rightorth::Alg₁=TensorKit.LQpos(),
                                alg_eigsolve::Alg₂=default_gauge_alg_eigsolve(tol, maxiter),
                                eig_miniter=10) where {TA,TAᵀ,TB,Alg₁,Alg₂}
        @assert all(isfullrank, A) "input tensors must be full rank"
        @assert domain(C) == codomain(C) "C must be square: $(space(C))"
        scalartype(A) === scalartype(C) ||
            throw(ArgumentError("A and C must have the same scalar type"))
        T = real(scalartype(A))
        return new{TA,TAᵀ,TB,T,Alg₁,Alg₂}(A, C, AR, AC_tail, CR,
                                          T(δ), T(tol), maxiter, verbosity,
                                          alg_rightorth, alg_eigsolve, eig_miniter)
    end
end

function _similar_tail(A::GenericMPSTensor)
    cod = _firstspace(A)
    dom = ⊗(dual(_lastspace(A)), dual.(space.(Ref(A), reverse(2:(numind(A) - 1))))...)
    return similar(A, cod ← dom)
end

function Base.iterate(it::RightGaugeIterable, iteration::Int=0)
    # check for termination
    if it.δ ≤ it.tol
        it.verbosity ≥ VERBOSE_CONVERGENCE &&
            @info "rightgauge converged after $iteration iterations: δ = $(it.δ)"
        return nothing
    elseif iteration ≥ it.maxiter
        it.verbosity ≥ VERBOSE_WARN &&
            @warn "rightgauge not converged after $iteration iterations: δ = $(it.δ)"
        return nothing
    end

    # eigsolve step
    if iteration ≥ it.eig_miniter
        # attempt to replicate previous code: tol = max(δ², tol / 10)
        alg_eigsolve = updatetol(it.alg_eigsolve, 1, it.δ^2)
        _, vecs = eigsolve(TransferMatrix(it.A, it.AR), it.C, 1, :LM, alg_eigsolve)
        it.C, _ = rightorth!(vecs[1]; alg=it.alg_rightorth)
    end

    # rightorth step
    it.CR[end] = it.C
    for i in length(it.AR):-1:1
        mul!(it.AR[i], it.A[i], it.CR[i]) # use AR as temporary storage for A-C
        _repartition!(it.AC_tail[i], it.AR[i])
        it.CR[i - 1], it.AC_tail[i] = rightorth!(it.AC_tail[i]; alg=it.alg_rightorth)
        # TODO: this last repartition is only strictly necessary for the last iteration
        # this is because AR is stored in the same format as AL.
        _repartition!(it.AR[i], it.AC_tail[i])
    end
    normalize!(it.CR[end])

    # check for convergence
    it.δ = norm(it.C - it.CR[end])
    it.C = it.CR[1]

    it.verbosity ≥ VERBOSE_ITER && @info "rightgauge iteration $iteration: δ = $(it.δ)"

    return it.C, iteration + 1
end

# ------------------------------------------------------------------------------------------
# Re-gauging AC and C
# ------------------------------------------------------------------------------------------

function regauge!(AC::PeriodicVector{<:GenericMPSTensor},
                  CR::PeriodicVector{<:MPSBondTensor}, alg::UniformGauging)
    if alg.order === :LR
        for i in 1:length(AC)
            # find AL that best fits these new AC and CR
            QAc, _ = leftorth!(AC[i]; alg=alg.alg_leftorth)
            Qc, _ = leftorth!(CR[i]; alg=alg.alg_leftorth)
            mul!(AC[i], QAc, Qc')
        end
    elseif alg.order === :RL
        for i in 1:length(AC)
            # find AR that best fits these new AC and CR
            AC_tail = _transpose_tail(AC[i])
            _, QAc = rightorth!(AC_tail; alg=alg.alg_rightorth)
            _, Qc = rightorth!(CR[i - 1]; alg=alg.alg_rightorth)
            mul!(AC_tail, Qc', QAc)
            _repartition!(AC[i], AC_tail)
        end
    else
        throw(ArgumentError("invalid order: $order"))
    end
    return AC
end

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
