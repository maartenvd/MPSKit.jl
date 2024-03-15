function approximate(ψ::InfiniteMPS,
                     toapprox::Tuple{<:Union{SparseMPO,DenseMPO},<:InfiniteMPS}, algorithm,
                     envs=environments(ψ, toapprox))
    # PeriodicMPO's always act on MPSMultiline's. To avoid code duplication, define everything in terms of MPSMultiline's.
    multi, envs = approximate(convert(MPSMultiline, ψ),
                              (convert(MPOMultiline, toapprox[1]),
                               convert(MPSMultiline, toapprox[2])), algorithm, envs)
    ψ = convert(InfiniteMPS, multi)
    return ψ, envs
end

Base.@deprecate(approximate(ψ::MPSMultiline, toapprox::Tuple{<:MPOMultiline,<:MPSMultiline},
                            alg::VUMPS, envs...; kwargs...),
                approximate(ψ, toapprox,
                            VOMPS(; alg.tol, alg.maxiter, alg.finalize,
                                  alg.verbosity, alg.alg_gauge, alg.alg_environments),
                            envs...; kwargs...))

function approximate(ψ::MPSMultiline, toapprox::Tuple{<:MPOMultiline,<:MPSMultiline},
                     alg::VOMPS, envs=environments(ψ, toapprox))
    ϵ::Float64 = calc_galerkin(ψ, envs)
    temp_ACs = similar.(ψ.AC)
    log = IterLog("VOMPS")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            @static if Defaults.parallelize_sites
                @sync for col in 1:size(ψ, 2)
                    Threads.@spawn _vomps_localupdate!(temp_ACs[:, col], col, ψ, toapprox,
                                                       envs)
                end
            else
                for col in 1:size(ψ, 2)
                    _vomps_localupdate!(temp_ACs[:, col], col, ψ, toapprox, envs)
                end
            end

            alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
            ψ = MPSMultiline(temp_ACs, ψ.CR[:, end]; alg_gauge.tol, alg_gauge.maxiter)

            alg_environments = updatetol(alg.alg_environments, iter, ϵ)
            recalculate!(envs, ψ; alg_environments.tol)

            ψ, envs = alg.finalize(iter, ψ, toapprox, envs)::typeof((ψ, envs))

            ϵ = calc_galerkin(ψ, envs)

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

    return ψ, envs, ϵ
end

function _vomps_localupdate!(AC′, loc, ψ, (O, ψ₀), envs, factalg=QRpos())
    local Q_AC, Q_C
    @static if Defaults.parallelize_sites
        @sync begin
            Threads.@spawn begin
                tmp_AC = circshift([ac_proj(row, loc, ψ, envs) for row in 1:size(ψ, 1)], 1)
                Q_AC = first.(leftorth!.(tmp_AC; alg=factalg))
            end
            Threads.@spawn begin
                tmp_C = circshift([c_proj(row, loc, ψ, envs) for row in 1:size(ψ, 1)], 1)
                Q_C = first.(leftorth!.(tmp_C; alg=factalg))
            end
        end
    else
        tmp_AC = circshift([ac_proj(row, loc, ψ, envs) for row in 1:size(ψ, 1)], 1)
        Q_AC = first.(leftorth!.(tmp_AC; alg=factalg))
        tmp_C = circshift([c_proj(row, loc, ψ, envs) for row in 1:size(ψ, 1)], 1)
        Q_C = first.(leftorth!.(tmp_C; alg=factalg))
    end
    return mul!.(AC′, Q_AC, adjoint.(Q_C))
end
