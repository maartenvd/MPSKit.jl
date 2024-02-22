#the statmech VUMPS
#it made sense to seperate both vumpses as
# - leading_boundary primarily works on MPSMultiline
# - they search for different eigenvalues
# - Hamiltonian vumps should use Lanczos, this has to use arnoldi
# - this vumps updates entire collumns (ψ[:,i]); incompatible with InfiniteMPS
# - (a)c-prime takes a different number of arguments
# - it's very litle duplicate code, but together it'd be a bit more convoluted (primarily because of the indexing way)

"""
    leading_boundary(ψ, opp, alg, envs=environments(ψ, opp))

Approximate the leading eigenvector for opp.
"""
function leading_boundary(ψ::InfiniteMPS, H, alg, envs=environments(ψ, H))
    (st, pr, de) = leading_boundary(convert(MPSMultiline, ψ), Multiline([H]), alg, envs)
    return convert(InfiniteMPS, st), pr, de
end

function leading_boundary(ψ::MPSMultiline, H, alg::VUMPS, envs=environments(ψ, H))
    galerkin = calc_galerkin(ψ, envs)
    iter = 1

    temp_ACs = similar.(ψ.AC)
    temp_Cs = similar.(ψ.CR)

    while true
        alg_eigsolve = updatetol(alg.alg_eigsolve, iter, galerkin)
        if Defaults.parallelize_sites
            @sync begin
                for col in 1:size(ψ, 2)
                    Threads.@spawn begin
                        H_AC = ∂∂AC($col, $ψ, $H, $envs)
                        ac = RecursiveVec($ψ.AC[:, col])
                        _, acvecs = eigsolve(H_AC, ac, 1, :LM, alg_eigsolve)
                        $temp_ACs[:, col] = acvecs[1].vecs[:]
                    end

                    Threads.@spawn begin
                        H_C = ∂∂C($col, $ψ, $H, $envs)
                        c = RecursiveVec($ψ.CR[:, col])
                        _, cvecs = eigsolve(H_C, c, 1, :LM, alg_eigsolve)
                        $temp_Cs[:, col] = cvecs[1].vecs[:]
                    end
                end
            end
        else
            for col in 1:size(ψ, 2)
                H_AC = ∂∂AC(col, ψ, H, envs)
                ac = RecursiveVec(ψ.AC[:, col])
                _, acvecs = eigsolve(H_AC, ac, 1, :LM, alg_eigsolve)
                temp_ACs[:, col] = acvecs[1].vecs[:]

                H_C = ∂∂C(col, ψ, H, envs)
                c = RecursiveVec(ψ.CR[:, col])
                _, cvecs = eigsolve(H_C, c, 1, :LM, alg_eigsolve)
                temp_Cs[:, col] = cvecs[1].vecs[:]
            end
        end

        for row in 1:size(ψ, 1), col in 1:size(ψ, 2)
            QAc, _ = leftorth!(temp_ACs[row, col]; alg=TensorKit.QRpos())
            Qc, _ = leftorth!(temp_Cs[row, col]; alg=TensorKit.QRpos())
            temp_ACs[row, col] = QAc * adjoint(Qc)
        end

        alg_gauge = updatetol(alg.alg_gauge, iter, galerkin)
        ψ = MPSMultiline(temp_ACs, ψ.CR[:, end]; alg_gauge.tol, alg_gauge.maxiter)

        alg_environments = updatetol(alg.alg_environments, iter, galerkin)
        recalculate!(envs, ψ; alg_environments.tol)

        ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ),typeof(envs)}

        galerkin = calc_galerkin(ψ, envs)
        alg.verbosity ≥ VERBOSE_ITER && @info "vumps @iteration $(iter) galerkin = $(galerkin)"

        if (galerkin <= alg.tol_galerkin) || iter >= alg.maxiter
            iter >= alg.maxiter && @warn "vumps didn't converge $(galerkin)"
            return ψ, envs, galerkin
        end

        iter += 1
    end
end
