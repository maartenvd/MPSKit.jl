"""
    DMRG{A,F} <: Algorithm

Single site DMRG algorithm for finding groundstates.

# Fields
- `tol::Float64`: tolerance for convergence criterium
- `eigalg::A`: eigensolver algorithm
- `maxiter::Int`: maximum number of outer iterations
- `verbosity::Int`: display progress information
- `finalize::F`: user-supplied function which is applied after each iteration, with
    signature `finalize(iter, ψ, H, envs) -> ψ, envs`
"""
struct DMRG{A,F} <: Algorithm
    tol::Float64
    maxiter::Int
    eigalg::A
    verbosity::Int
    finalize::F
end
function DMRG(; tol::Real=Defaults.tol, maxiter::Integer=Defaults.maxiter,
              finalize=Defaults._finalize, eigalg=Defaults.eigsolver,
              verbose=nothing, verbosity::Integer=Defaults.verbosity)
    # Deprecation warnings
    actual_verbosity = if !isnothing(verbose)
        Base.depwarn("DMRG(; kwargs..., verbose=...) is deprecated. Use DMRG(; kwargs..., verbosity=...) instead.",
                     :DMRG; force=true)
        verbose ? Iteration : Warning
    else
        verbosity
    end
    return DMRG{typeof(eigalg),typeof(finalize)}(tol, maxiter, eigalg, actual_verbosity,
                                                 finalize)
end

function find_groundstate!(ψ::AbstractFiniteMPS, H, alg::DMRG, envs=environments(ψ, H))
    t₀ = Base.time_ns()
    ϵ::Float64 = 2 * alg.tol

    for iter in 1:(alg.maxiter)
        global ϵ = 0.0
        Δt = @elapsed begin
            for pos in [1:(length(ψ) - 1); length(ψ):-1:2]
                h = ∂∂AC(pos, ψ, H, envs)
                _, vecs = eigsolve(h, ψ.AC[pos], 1, :SR, alg.eigalg)
                ϵ = max(ϵ, calc_galerkin(ψ, pos, envs))
                ψ.AC[pos] = vecs[1]
            end

            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ),typeof(envs)}
        end

        alg.verbosity >= Iteration &&
            @info "DMRG iteration:" iter ϵ λ = sum(expectation_value(ψ, H, envs)) Δt

        ϵ <= alg.tol && break
        alg.verbosity >= Warning && iter == alg.maxiter &&
            @warn "DMRG maximum iterations" iter ϵ λ = sum(expectation_value(ψ, H, envs))
    end

    Δt = (Base.time_ns() - t₀) / 1.0e9
    alg.verbosity >= Convergence &&
        @info "DMRG summary:" ϵ λ = sum(expectation_value(ψ, H, envs)) Δt
    return ψ, envs, ϵ
end

"""
    DMRG2{A,F} <: Algorithm

2-site  DMRG algorithm for finding groundstates.

# Fields
- `tol::Float64`: tolerance for convergence criterium
- `eigalg::A`: eigensolver algorithm
- `maxiter::Int`: maximum number of outer iterations
- `verbosity::Int`: display progress information
- `finalize::F`: user-supplied function which is applied after each iteration, with
    signature `finalize(iter, ψ, H, envs) -> ψ, envs`
- `trscheme`: truncation algorithm for [tsvd][TensorKit.tsvd](@ref)
"""
@kwdef struct DMRG2{A,F} <: Algorithm
    tol::Float64
    maxiter::Int
    eigalg::A
    trscheme::TruncationScheme
    verbosity
    finalize::F
end
function DMRG2(; tol::Real = Defaults.tol, maxiter::Integer = Defaults.maxiter,
               finalize = Defaults._finalize, eigalg = Defaults.eigsolver,
               trscheme = truncerr(sqrt(tol)), verbose = nothing, verbosity::Integer = Defaults.verbosity)
    # Deprecation warnings
    actual_verbosity = if !isnothing(verbose)
        Base.depwarn("DMRG2(; kwargs..., verbose=...) is deprecated. Use DMRG2(; kwargs..., verbosity=...) instead.",
                     :DMRG2; force=true)
        verbose ? Iteration : Warning
    else
        verbosity
    end
    return DMRG2{typeof(eigalg),typeof(finalize)}(tol, maxiter, eigalg, trscheme, actual_verbosity,
                                                 finalize)
end

function find_groundstate!(ψ::AbstractFiniteMPS, H, alg::DMRG2, envs=environments(ψ, H))
    t₀ = Base.time_ns()
    ϵ::Float64 = 2 * alg.tol

    for iter in 1:(alg.maxiter)
        ϵ = 0.0
        Δt = @elapsed begin
            #left to right sweep
            for pos in 1:(length(ψ) - 1)
                @plansor ac2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos + 1][1 -4; -3]

                _, vecs = eigsolve(∂∂AC2(pos, ψ, H, envs), ac2, 1, :SR, alg.eigalg)
                newA2center = first(vecs)

                al, c, ar, = tsvd!(newA2center; trunc=alg.trscheme, alg=TensorKit.SVD())
                normalize!(c)
                v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) *
                             conj(ar[6; 3 4])
                ϵ = max(ϵ, abs(1 - abs(v)))

                ψ.AC[pos] = (al, complex(c))
                ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
            end

            for pos in (length(ψ) - 2):-1:1
                @plansor ac2[-1 -2; -3 -4] := ψ.AL[pos][-1 -2; 1] * ψ.AC[pos + 1][1 -4; -3]

                _, vecs = eigsolve(∂∂AC2(pos, ψ, H, envs), ac2, 1, :SR, alg.eigalg)
                newA2center = first(vecs)

                al, c, ar, = tsvd!(newA2center; trunc=alg.trscheme, alg=TensorKit.SVD())
                normalize!(c)
                v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) *
                             conj(ar[6; 3 4])
                ϵ = max(ϵ, abs(1 - abs(v)))

                ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
                ψ.AC[pos] = (al, complex(c))
            end

            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ),typeof(envs)}
        end

        alg.verbosity >= Iteration &&
            @info "DMRG2 iteration:" iter ϵ λ = sum(expectation_value(ψ, H, envs)) Δt

        ϵ <= alg.tol && break
        alg.verbosity >= Warning && iter == alg.maxiter &&
            @warn "DMRG2 maximum iterations" iter ϵ λ = sum(expectation_value(ψ, H, envs))
    end

    Δt = (Base.time_ns() - t₀) / 1.0e9
    alg.verbosity >= Convergence && @info "DMRG2 summary:" ϵ λ = sum(expectation_value(ψ, H, envs)) Δt
    return ψ, envs, ϵ
end

function find_groundstate(ψ, H, alg::Union{DMRG,DMRG2}, envs...)
    return find_groundstate!(copy(ψ), H, alg, envs...)
end
