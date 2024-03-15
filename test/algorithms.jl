println("
------------------
|   Algorithms   |
------------------
")
module TestAlgorithms

using ..TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

@testset "find_groundstate" verbose = true begin
    tol = 1e-8
    verbosity = 3
    infinite_algs = [VUMPS(; tol, verbosity),
                     IDMRG1(; tol, verbosity),
                     IDMRG2(; trscheme=truncdim(12), tol, verbosity),
                     GradientGrassmann(; tol, verbosity),
                     VUMPS(; tol=100 * tol, verbosity) &
                     GradientGrassmann(; tol, verbosity)]

    g = 4.0
    D = 6
    H1 = force_planar(transverse_field_ising(; g))

    @testset "Infinite $i" for (i, alg) in enumerate(infinite_algs[1:2])
        L = alg isa IDMRG2 ? 2 : 1
        ψ₀ = repeat(InfiniteMPS([ℙ^2], [ℙ^D]), L)
        H = repeat(H1, L)

        v₀ = variance(ψ₀, H)
        ψ, envs, δ = find_groundstate(ψ₀, H, alg)
        v = variance(ψ, H, envs)

        @test sum(δ) < 1e-3
        @test v₀ > v && v < 1e-2 # energy variance should be low
    end

    # Hlazy1 = LazySum([3 * H1, -1 * H1, 5.557 * H1])

    # @testset "LazySum Infinite $i" for (i, alg) in enumerate(infinite_algs)
    #     L = alg isa IDMRG2 ? 2 : 1
    #     ψ₀ = repeat(InfiniteMPS([ℙ^2], [ℙ^D]), L)
    #     Hlazy = repeat(Hlazy1, L)

    #     v₀ = variance(ψ₀, Hlazy)
    #     ψ, envs, δ = find_groundstate(ψ₀, Hlazy, alg)
    #     v = variance(ψ, Hlazy)

    #     @test sum(δ) < 1e-3
    #     @test v₀ > v && v < 1e-2 # energy variance should be low

    #     ψ_nolazy, envs_nolazy, _ = find_groundstate(ψ₀, sum(Hlazy), alg)
    #     @test expectation_value(ψ, Hlazy,
    #                             envs) ≈
    #           expectation_value(ψ_nolazy, sum(Hlazy), envs_nolazy) atol = 1 - 06
    # end

    # finite_algs = [DMRG(; verbosity),
    #                DMRG2(; verbosity, trscheme=truncdim(D)),
    #                GradientGrassmann(; tol, verbosity, maxiter=300)]

    # H = force_planar(transverse_field_ising(; g))

    # @testset "Finite $i" for (i, alg) in enumerate(finite_algs)
    #     ψ₀ = FiniteMPS(rand, ComplexF64, 10, ℙ^2, ℙ^D)

    #     v₀ = variance(ψ₀, H)
    #     ψ, envs, δ = find_groundstate(ψ₀, H, alg)
    #     v = variance(ψ, H, envs)

    #     @test sum(δ) < 1e-3
    #     @test v₀ > v && v < 1e-2 # energy variance should be low
    # end

    # Hlazy = LazySum([3 * H, H, 5.557 * H])

    # @testset "LazySum Finite $i" for (i, alg) in enumerate(finite_algs)
    #     ψ₀ = FiniteMPS(rand, ComplexF64, 10, ℙ^2, ℙ^D)

    #     v₀ = variance(ψ₀, Hlazy)
    #     ψ, envs, δ = find_groundstate(ψ₀, Hlazy, alg)
    #     v = variance(ψ, Hlazy)

    #     @test sum(δ) < 1e-3
    #     @test v₀ > v && v < 1e-2 # energy variance should be low

    #     ψ_nolazy, envs_nolazy, _ = find_groundstate(ψ₀, sum(Hlazy), alg)
    #     @test expectation_value(ψ, Hlazy, envs) ≈
    #           expectation_value(ψ_nolazy, sum(Hlazy), envs_nolazy) atol = 1 - 06
    # end
end

@testset "timestep" verbose = true begin
    dt = 0.1
    algs = [TDVP(), TDVP2()]

    H = force_planar(heisenberg_XXX(; spin=1 // 2))
    ψ₀ = FiniteMPS(fill(TensorMap(rand, ComplexF64, ℙ^1 * ℙ^2, ℙ^1), 5))
    E₀ = expectation_value(ψ₀, H)

    @testset "Finite $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
        ψ, envs = timestep(ψ₀, H, 0.0, dt, alg)
        E = expectation_value(ψ, H, envs)
        @test sum(E₀) ≈ sum(E) atol = 1e-2
    end

    Hlazy = LazySum([3 * H, 1.55 * H, -0.1 * H])

    @testset "Finite LazySum $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
        ψ, envs = timestep(ψ₀, Hlazy, 0.0, dt, alg)
        E = expectation_value(ψ, Hlazy, envs)
        @test (3 + 1.55 - 0.1) * sum(E₀) ≈ sum(E) atol = 1e-2
    end

    Ht = MultipliedOperator(H, t -> 4) + MultipliedOperator(H, 1.45)

    @testset "Finite TimeDependent LazySum $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in
                                                                                   algs
        ψ, envs = timestep(ψ₀, Ht(1.0), 0.0, dt, alg)
        E = expectation_value(ψ, Ht(1.0), envs)

        ψt, envst = timestep(ψ₀, Ht, 1.0, dt, alg)
        Et = expectation_value(ψt, Ht(1.0), envst)
        @test sum(E) ≈ sum(Et) atol = 1e-8
    end

    H = repeat(force_planar(heisenberg_XXX(; spin=1)), 2)
    ψ₀ = InfiniteMPS([ℙ^3, ℙ^3], [ℙ^50, ℙ^50])
    E₀ = expectation_value(ψ₀, H)

    @testset "Infinite TDVP" begin
        ψ, envs = timestep(ψ₀, H, 0.0, dt, TDVP())
        E = expectation_value(ψ, H, envs)
        @test sum(E₀) ≈ sum(E) atol = 1e-2
    end

    Hlazy = LazySum([3 * H, 1.55 * H, -0.1 * H])

    @testset "Infinite LazySum TDVP" begin
        ψ, envs = timestep(ψ₀, Hlazy, 0.0, dt, TDVP())
        E = expectation_value(ψ, Hlazy, envs)
        @test (3 + 1.55 - 0.1) * sum(E₀) ≈ sum(E) atol = 1e-2
    end

    Ht = MultipliedOperator(H, t -> 4) + MultipliedOperator(H, 1.45)

    @testset "Infinite TimeDependent LazySum" begin
        ψ, envs = timestep(ψ₀, Ht(1.0), 0.0, dt, TDVP())
        E = expectation_value(ψ, Ht(1.0), envs)

        ψt, envst = timestep(ψ₀, Ht, 1.0, dt, TDVP())
        Et = expectation_value(ψt, Ht(1.0), envst)
        @test sum(E) ≈ sum(Et) atol = 1e-8
    end
end

@testset "time_evolve" verbose = true begin
    t_span = 0:0.1:0.1
    algs = [TDVP(), TDVP2()]

    H = force_planar(heisenberg_XXX(; spin=1 // 2))
    ψ₀ = FiniteMPS(fill(TensorMap(rand, ComplexF64, ℙ^1 * ℙ^2, ℙ^1), 5))
    E₀ = expectation_value(ψ₀, H)

    @testset "Finite $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
        ψ, envs = time_evolve(ψ₀, H, t_span, alg)
        E = expectation_value(ψ, H, envs)
        @test sum(E₀) ≈ sum(E) atol = 1e-2
    end

    H = repeat(force_planar(heisenberg_XXX(; spin=1)), 2)
    ψ₀ = InfiniteMPS([ℙ^3, ℙ^3], [ℙ^50, ℙ^50])
    E₀ = expectation_value(ψ₀, H)

    @testset "Infinite TDVP" begin
        ψ, envs = time_evolve(ψ₀, H, t_span, TDVP())
        E = expectation_value(ψ, H, envs)
        @test sum(E₀) ≈ sum(E) atol = 1e-2
    end
end

@testset "leading_boundary" verbose = true begin
    tol = 1e-5
    verbosity = 0
    algs = [VUMPS(; tol, verbosity), VOMPS(; tol, verbosity),
            GradientGrassmann(; verbosity)]
    mpo = force_planar(classical_ising())

    ψ₀ = InfiniteMPS([ℙ^2], [ℙ^10])
    @testset "Infinite $i" for (i, alg) in enumerate(algs)
        ψ, envs = leading_boundary(ψ₀, mpo, alg)
        ψ, envs = changebonds(ψ, mpo, OptimalExpand(; trscheme=truncdim(3)), envs)
        ψ, envs = leading_boundary(ψ, mpo, alg)

        @test dim(space(ψ.AL[1, 1], 1)) == dim(space(ψ₀.AL[1, 1], 1)) + 3
        @test expectation_value(ψ, envs)[1, 1] ≈ 2.5337 atol = 1e-3
    end
end

@testset "quasiparticle_excitation" verbose = true begin
    @testset "infinite (ham)" begin
        H = repeat(force_planar(heisenberg_XXX()), 2)
        ψ = InfiniteMPS([ℙ^3, ℙ^3], [ℙ^48, ℙ^48])
        ψ, envs, _ = find_groundstate(ψ, H; maxiter=400, verbosity=0, tol=1e-11)
        energies, ϕs = excitations(H, QuasiparticleAnsatz(), Float64(pi), ψ, envs)
        @test energies[1] ≈ 0.41047925 atol = 1e-4
        @test variance(ϕs[1], H) < 1e-8
    end
    @testset "infinite (mpo)" begin
        H = repeat(sixvertex(), 2)
        ψ = InfiniteMPS([ℂ^2, ℂ^2], [ℂ^10, ℂ^10])
        ψ, envs, _ = leading_boundary(ψ, H, VUMPS(; maxiter=400, verbosity=0))
        energies, ϕs = excitations(H, QuasiparticleAnsatz(), [0.0, Float64(pi / 2)], ψ,
                                   envs; verbosity=0)
        @test abs(energies[1]) > abs(energies[2]) # has a minima at pi/2
    end

    @testset "finite" begin
        verbosity = 0
        H = force_planar(transverse_field_ising())
        ψ = InfiniteMPS([ℙ^2], [ℙ^10])
        ψ, envs, _ = find_groundstate(ψ, H; maxiter=400, verbosity, tol=1e-9)
        energies, ϕs = excitations(H, QuasiparticleAnsatz(), 0.0, ψ, envs)
        inf_en = energies[1]

        fin_en = map([20, 10]) do len
            ψ = FiniteMPS(rand, ComplexF64, len, ℙ^2, ℙ^15)
            (ψ, envs, _) = find_groundstate(ψ, H; verbosity)

            #find energy with quasiparticle ansatz
            energies_QP, ϕs = excitations(H, QuasiparticleAnsatz(), ψ, envs)
            @test variance(ϕs[1], H) < 1e-6

            #find energy with normal dmrg
            energies_dm, _ = excitations(H,
                                         FiniteExcited(;
                                                       gsalg=DMRG(; verbosity,
                                                                  tol=1e-6)), ψ)
            @test energies_dm[1] ≈ energies_QP[1] + sum(expectation_value(ψ, H, envs)) atol = 1e-4

            return energies_QP[1]
        end

        @test issorted(abs.(fin_en .- inf_en))
    end
end

@testset "changebonds $((pspace,Dspace))" verbose = true for (pspace, Dspace) in
                                                             [(ℙ^4, ℙ^3),
                                                              (Rep[SU₂](1 => 1),
                                                               Rep[SU₂](0 => 2, 1 => 2,
                                                                        2 => 1))]
    @testset "mpo" begin
        #random nn interaction
        nn = TensorMap(rand, ComplexF64, pspace * pspace, pspace * pspace)
        nn += nn'

        mpo1 = periodic_boundary_conditions(convert(DenseMPO,
                                                    make_time_mpo(MPOHamiltonian(nn), 0.1,
                                                                  WII())), 10)
        mpo2 = changebonds(mpo1, SvdCut(; trscheme=truncdim(5)))

        @test dim(space(mpo2[5], 1)) < dim(space(mpo1[5], 1))
    end

    @testset "infinite mps" begin
        #random nn interaction
        nn = TensorMap(rand, ComplexF64, pspace * pspace, pspace * pspace)
        nn += nn'

        state = InfiniteMPS([pspace, pspace], [Dspace, Dspace])

        state_re = changebonds(state,
                               RandExpand(; trscheme=truncdim(dim(Dspace) * dim(Dspace))))
        @test dot(state, state_re) ≈ 1 atol = 1e-8

        state_oe, _ = changebonds(state,
                                  repeat(MPOHamiltonian(nn), 2),
                                  OptimalExpand(;
                                                trscheme=truncdim(dim(Dspace) *
                                                                  dim(Dspace))))
        @test dot(state, state_oe) ≈ 1 atol = 1e-8

        state_vs, _ = changebonds(state, repeat(MPOHamiltonian(nn), 2),
                                  VUMPSSvdCut(; trscheme=notrunc()))
        @test dim(left_virtualspace(state, 1)) < dim(left_virtualspace(state_vs, 1))

        state_vs_tr = changebonds(state_vs, SvdCut(; trscheme=truncdim(dim(Dspace))))
        @test dim(right_virtualspace(state_vs_tr, 1)) < dim(right_virtualspace(state_vs, 1))
    end

    @testset "finite mps" begin
        #random nn interaction
        nn = TensorMap(rand, ComplexF64, pspace * pspace, pspace * pspace)
        nn += nn'

        state = FiniteMPS(10, pspace, Dspace)

        state_re = changebonds(state,
                               RandExpand(; trscheme=truncdim(dim(Dspace) * dim(Dspace))))
        @test dot(state, state_re) ≈ 1 atol = 1e-8

        state_oe, _ = changebonds(state, MPOHamiltonian(nn),
                                  OptimalExpand(;
                                                trscheme=truncdim(dim(Dspace) * dim(Dspace))))
        @test dot(state, state_oe) ≈ 1 atol = 1e-8

        state_tr = changebonds(state_oe, SvdCut(; trscheme=truncdim(dim(Dspace))))

        @test dim(left_virtualspace(state_tr, 5)) < dim(right_virtualspace(state_oe, 5))
    end

    @testset "MPSMultiline" begin
        o = TensorMap(rand, ComplexF64, pspace * pspace, pspace * pspace)
        mpo = MPOMultiline(o)

        t = TensorMap(rand, ComplexF64, Dspace * pspace, Dspace)
        state = MPSMultiline(fill(t, 1, 1))

        state_re = changebonds(state,
                               RandExpand(; trscheme=truncdim(dim(Dspace) * dim(Dspace))))
        @test dot(state, state_re) ≈ 1 atol = 1e-8

        state_oe, _ = changebonds(state, mpo,
                                  OptimalExpand(;
                                                trscheme=truncdim(dim(Dspace) *
                                                                  dim(Dspace))))
        @test dot(state, state_oe) ≈ 1 atol = 1e-8

        state_tr = changebonds(state_oe, SvdCut(; trscheme=truncdim(dim(Dspace))))

        @test dim(right_virtualspace(state_tr, 1, 1)) <
              dim(left_virtualspace(state_oe, 1, 1))
    end
end

@testset "Dynamical DMRG" verbose = true begin
    ham = force_planar(-1.0 * transverse_field_ising(; g=-4.0))
    gs, = find_groundstate(InfiniteMPS([ℙ^2], [ℙ^10]), ham, VUMPS(; verbosity=0))
    window = WindowMPS(gs, copy.([gs.AC[1]; [gs.AR[i] for i in 2:10]]), gs)

    szd = force_planar(TensorMap(ComplexF64[0.5 0; 0 -0.5], ℂ^2 ← ℂ^2))
    @test expectation_value(gs, szd)[1] ≈ expectation_value(window, szd)[1] atol = 1e-10

    polepos = expectation_value(gs, ham, 10)
    @test polepos ≈ expectation_value(window, ham)[2]

    vals = (-0.5:0.2:0.5) .+ polepos
    eta = 0.3im

    predicted = [1 / (v + eta - polepos) for v in vals]

    @testset "Flavour $f" for f in (Jeckelmann(), NaiveInvert())
        alg = DynamicalDMRG(; flavour=f, verbosity=0, tol=1e-8)
        data = map(vals) do v
            result, = propagator(window, v + eta, ham, alg)
            return result
        end
        @test data ≈ predicted atol = 1e-8
    end
end

@testset "fidelity susceptibility" begin
    X = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)

    H_X = MPOHamiltonian(X)
    H_ZZ = MPOHamiltonian(Z ⊗ Z)

    hamiltonian(λ) = H_ZZ + λ * H_X
    analytical_susceptibility(λ) = abs(1 / (16 * λ^2 * (λ^2 - 1)))

    for λ in [1.05, 2.0, 4.0]
        H = hamiltonian(λ)
        ψ = InfiniteMPS([ℂ^2], [ℂ^16])
        ψ, envs, = find_groundstate(ψ, H, VUMPS(; maxiter=100, verbosity=0))

        numerical_scusceptibility = fidelity_susceptibility(ψ, H, [H_X], envs; maxiter=10)
        @test numerical_scusceptibility[1, 1] ≈ analytical_susceptibility(λ) atol = 1e-2

        # test if the finite fid sus approximates the analytical one with increasing system size
        fin_en = map([20, 15, 10]) do L
            ψ = FiniteMPS(rand, ComplexF64, L, ℂ^2, ℂ^16)
            ψ, envs, = find_groundstate(ψ, H, DMRG(; verbosity=0))
            numerical_scusceptibility = fidelity_susceptibility(ψ, H, [H_X], envs;
                                                                maxiter=10)
            return numerical_scusceptibility[1, 1] / L
        end
        @test issorted(abs.(fin_en .- analytical_susceptibility(λ)))
    end
end

#stub tests
@testset "correlation length / entropy" begin
    st = InfiniteMPS([ℙ^2], [ℙ^10])
    th = force_planar(transverse_field_ising())
    (st, _) = find_groundstate(st, th, VUMPS(; verbosity=0))
    len_crit = correlation_length(st)[1]
    entrop_crit = entropy(st)

    th = force_planar(transverse_field_ising(; g=4))
    (st, _) = find_groundstate(st, th, VUMPS(; verbosity=0))
    len_gapped = correlation_length(st)[1]
    entrop_gapped = entropy(st)

    @test len_crit > len_gapped
    @test real(entrop_crit) > real(entrop_gapped)
end

@testset "expectation value / correlator" begin
    st = InfiniteMPS([ℂ^2], [ℂ^10])
    th = transverse_field_ising(; g=4)
    (st, _) = find_groundstate(st, th, VUMPS(; verbosity=0))

    sz_mpo = TensorMap([1.0 0; 0 -1], ℂ^1 * ℂ^2, ℂ^2 * ℂ^1)
    sz = TensorMap([1.0 0; 0 -1], ℂ^2, ℂ^2)
    id_mpo = TensorMap([1.0 0; 0 1.0], ℂ^1 * ℂ^2, ℂ^2 * ℂ^1)
    @tensor szsz[-1 -2; -3 -4] := sz[-1 -3] * sz[-2 -4]

    @test isapprox(expectation_value(st, [sz_mpo], 1), expectation_value(st, sz, 1),
                   atol=1e-2)
    @test isapprox(expectation_value(st, [sz_mpo, sz_mpo], 1),
                   expectation_value(st, szsz, 1),
                   atol=1e-2)
    @test isapprox(expectation_value(st, [sz_mpo, sz_mpo], 2),
                   expectation_value(st, szsz, 1),
                   atol=1e-2)

    G = correlator(st, sz_mpo, sz_mpo, 1, 2:5)
    G2 = correlator(st, szsz, 1, 3:2:5)
    @test isapprox(G[2], G2[1], atol=1e-2)
    @test isapprox(last(G), last(G2), atol=1e-2)
    @test isapprox(G[1], expectation_value(st, szsz, 1), atol=1e-2)
    @test isapprox(G[2], expectation_value(st, [sz_mpo, id_mpo, sz_mpo], 1), atol=1e-2)
    @test isapprox(first(correlator(st, sz_mpo, sz_mpo, 1, 2)),
                   expectation_value(st, szsz, 1),
                   atol=1e-2)
end

@testset "approximate" verbose = true begin
    verbosity = 0
    @testset "mpo * infinite ≈ infinite" begin
        st = InfiniteMPS([ℙ^2, ℙ^2], [ℙ^10, ℙ^10])
        th = force_planar(repeat(transverse_field_ising(; g=4), 2))

        dt = 1e-3
        sW1 = make_time_mpo(th, dt, TaylorCluster{3}())
        sW2 = make_time_mpo(th, dt, WII())
        W1 = convert(DenseMPO, sW1)
        W2 = convert(DenseMPO, sW2)

        st1, _ = approximate(st, (sW1, st), VOMPS(; verbosity))
        st2, _ = approximate(st, (W2, st), VOMPS(; verbosity))
        st3, _ = approximate(st, (W1, st), IDMRG1(; verbosity))
        st4, _ = approximate(st, (sW2, st), IDMRG2(; trscheme=truncdim(20), verbosity))
        st5, _ = timestep(st, th, 0.0, dt, TDVP())
        st6 = changebonds(W1 * st, SvdCut(; trscheme=truncdim(10)))

        @test abs(dot(st1, st5)) ≈ 1.0 atol = dt
        @test abs(dot(st3, st5)) ≈ 1.0 atol = dt
        @test abs(dot(st6, st5)) ≈ 1.0 atol = dt
        @test abs(dot(st2, st4)) ≈ 1.0 atol = dt

        nW1 = changebonds(W1, SvdCut(; trscheme=truncerr(dt))) #this should be a trivial mpo now
        @test dim(space(nW1.opp[1, 1], 1)) == 1
    end

    finite_algs = [DMRG(; verbosity), DMRG2(; verbosity, trscheme=truncdim(10))]
    @testset "finitemps1 ≈ finitemps2" for alg in finite_algs
        a = FiniteMPS(10, ℂ^2, ℂ^10)
        b = FiniteMPS(10, ℂ^2, ℂ^20)

        before = abs(dot(a, b))

        a = first(approximate(a, b, alg))

        after = abs(dot(a, b))

        @test before < after
    end

    @testset "sparse_mpo * finitemps1 ≈ finitemps2" for alg in finite_algs
        ψ₁ = FiniteMPS(10, ℂ^2, ℂ^30)
        ψ₂ = FiniteMPS(10, ℂ^2, ℂ^25)

        H = transverse_field_ising(; g=4.0)
        τ = 1e-3

        expH = make_time_mpo(H, τ, WI())
        ψ₂, = approximate(ψ₂, (expH, ψ₁), alg)
        normalize!(ψ₂)
        ψ₂′, = timestep(ψ₁, H, 0.0, τ, TDVP())
        @test abs(dot(ψ₁, ψ₁)) ≈ abs(dot(ψ₂, ψ₂′)) atol = 0.001
    end

    @testset "dense_mpo * finitemps1 ≈ finitemps2" for alg in finite_algs
        Ψ₁ = FiniteMPS(10, ℂ^2, ℂ^20)
        Ψ₂ = FiniteMPS(10, ℂ^2, ℂ^10)

        O = finite_classical_ising(10)
        Ψ₂, = approximate(Ψ₂, (O, Ψ₁), alg)

        @test norm(O * Ψ₁ - Ψ₂) ≈ 0 atol = 0.001
    end
end

@testset "periodic boundary conditions" begin
    len = 10

    #impose periodic boundary conditions on the hamiltonian (circle size 10)
    th = transverse_field_ising()
    th = periodic_boundary_conditions(th, len)

    ψ = FiniteMPS(len, ℂ^2, ℂ^10)

    gs, envs = find_groundstate(ψ, th, DMRG(; verbosity=0))

    #translation mpo:
    @tensor bulk[-1 -2; -3 -4] := isomorphism(ℂ^2, ℂ^2)[-2, -4] *
                                  isomorphism(ℂ^2, ℂ^2)[-1, -3]
    translation = periodic_boundary_conditions(DenseMPO(bulk), len)

    #the groundstate should be translation invariant:
    ut = Tensor(ones, ℂ^1)
    @tensor leftstart[-1 -2; -3] := l_LL(gs)[-1, -3] * conj(ut[-2])
    T = TransferMatrix([gs.AC[1]; gs.AR[2:end]], translation[:], [gs.AC[1]; gs.AR[2:end]])
    v = leftstart * T

    expval = @tensor v[1, 2, 3] * r_RR(gs)[3, 1] * ut[2]

    @test expval ≈ 1 atol = 1e-5

    energies, values = exact_diagonalization(th; which=:SR)
    @test energies[1] ≈ sum(expectation_value(gs, th)) atol = 1e-5
end

end
