"""
    struct VUMPSSvdCut <: Algorithm end

An algorithm that uses an two-site VUMPS step to change the bond dimension of a state.

# Fields
- `tol::Real = Defaults.tol` : The tolerance for the eigenvalue solver.
- `tol_gauge::Real = Defaults.tolgauge` : The tolerance for the gauge.
- `tol::Real = Defaults.tol` : The tolerance for the Galerkin truncation.
- `tol_eigenval::Real = Defaults.tol` : The tolerance for the eigenvalue solver.
- `trscheme::TruncationScheme = notrunc()` : The truncation scheme to use.
"""
@kwdef struct VUMPSSvdCut <: Algorithm
    tol_gauge = Defaults.tolgauge
    tol = Defaults.tol
    tol_eigenval = Defaults.tol
    trscheme = notrunc()
end

function changebonds_1(ψ₁::InfiniteMPS, H, alg::VUMPSSvdCut,
                       envs=environments(ψ₁, H)) # would be more efficient if we also repeated envs
    # the unitcell==1 case is unique, because there you have a sef-consistency condition

    # expand the one site to two sites
    ψ₂ = InfiniteMPS(repeat(ψ₁.AL, 2))
    H₂ = repeat(H, 2)

    ψ₂, nenvs = changebonds(ψ₂, H₂, alg)

    D1 = space(ψ₂.AL[1], 1)
    D2 = space(ψ₂.AL[2], 1)

    # collapse back to 1 site
    if D2 != D1
        trscheme = truncspace(infimum(D1, D2))
        ψ₂, nenvs = changebonds(ψ₂, H₂, SvdCut(; trscheme), nenvs)
    end

    ψ′ = InfiniteMPS([ψ₂.AL[1]], ψ₂.CR[1]; tol=alg.tol_gauge)
    return ψ′, envs
end

function changebonds_n(ψ::InfiniteMPS, H, alg::VUMPSSvdCut, envs=environments(ψ, H))
    meps = 0.0
    for loc in 1:length(ψ)
        @plansor AC2[-1 -2; -3 -4] := ψ.AC[loc][-1 -2; 1] * ψ.AR[loc + 1][1 -4; -3]

        h_ac2 = ∂∂AC2(loc, ψ, H, envs)
        vals, vecs, _ = eigsolve(h_ac2, AC2, 1, :SR; tol=alg.tol,
                                 ishermitian=false)
        nAC2 = vecs[1]

        h_c = ∂∂C(loc + 1, ψ, H, envs)
        vals, vecs, _ = eigsolve(h_c, ψ.CR[loc + 1], 1, :SR; tol=alg.tol,
                                 ishermitian=false)
        nC2 = vecs[1]

        # svd ac2, get new AL1 and S,V --> AC
        AL1, S, V, eps = tsvd(nAC2; trunc=alg.trscheme, alg=TensorKit.SVD())
        @plansor AC[-1 -2; -3] := S[-1; 1] * V[1; -3 -2]
        meps = max(eps, meps)

        # find AL2 from AC and C as in vumps paper
        QAC, _ = leftorth(AC; alg=QRpos())
        QC, _ = leftorth(nC2; alg=QRpos())
        dom_map = isometry(domain(QC), domain(QAC))

        @plansor AL2[-1 -2; -3] := QAC[-1 -2; 1] * conj(dom_map[2; 1]) * conj(QC[-3; 2])

        # make a new ψ using the updated A's
        copied = copy(ψ.AL)
        copied[loc] = AL1
        copied[loc + 1] = AL2
        ψ = InfiniteMPS(copied; tol=alg.tol_gauge)
    end

    return ψ, envs
end

function changebonds(ψ::InfiniteMPS, H, alg::VUMPSSvdCut, envs=environments(ψ, H))
    return if isone(length(ψ))
        changebonds_1(ψ, H, alg, envs)
    else
        changebonds_n(ψ, H, alg, envs)
    end
end
