"""
    TDVP{A} <: Algorithm

Single site [TDVP](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.070601)
algorithm for time evolution.

# Fields
- `integrator::A`: integration algorithm (defaults to Lanczos exponentiation)
- `tolgauge::Float64`: tolerance for gauging algorithm
- `gaugemaxiter::Int`: maximum amount of gauging iterations
- `finalize::F`: user-supplied function which is applied after each timestep, with
    signature `finalize(t, Ψ, H, envs) -> Ψ, envs`
"""
@kwdef struct TDVP{A,F} <: Algorithm
    integrator::A = Defaults.alg_expsolve()
    tolgauge::Float64 = Defaults.tolgauge
    gaugemaxiter::Int = Defaults.maxiter
    finalize::F = Defaults._finalize
end

function timestep(Ψ::InfiniteMPS, H, t::Number, dt::Number, alg::TDVP,
                  envs::Union{Cache,MultipleEnvironments}=environments(Ψ, H);
                  leftorthflag=true)
    temp_ACs = similar(Ψ.AC)
    temp_CRs = similar(Ψ.CR)
    @sync for (loc, (ac, c)) in enumerate(zip(Ψ.AC, Ψ.CR))
        Threads.@spawn begin
            h_ac = ∂∂AC(loc, Ψ, H, envs)
            temp_ACs[loc] = integrate(h_ac, ac, t, dt, alg.integrator)
        end

        Threads.@spawn begin
            h_c = ∂∂C(loc, Ψ, H, envs)
            temp_CRs[loc] = integrate(h_c, c, t, dt, alg.integrator)
        end
    end

    if leftorthflag
        for loc in 1:length(Ψ)
            # find AL that best fits these new Acenter and centers
            QAc, _ = leftorth!(temp_ACs[loc]; alg=TensorKit.QRpos())
            Qc, _ = leftorth!(temp_CRs[loc]; alg=TensorKit.QRpos())
            @plansor temp_ACs[loc][-1 -2; -3] = QAc[-1 -2; 1] * conj(Qc[-3; 1])
        end
        newΨ = InfiniteMPS(temp_ACs, Ψ.CR[end]; tol=alg.tolgauge, maxiter=alg.gaugemaxiter)

    else
        for loc in 1:length(Ψ)
            # find AR that best fits these new Acenter and centers
            _, QAc = rightorth!(_transpose_tail(temp_ACs[loc]); alg=TensorKit.LQpos())
            _, Qc = rightorth!(temp_CRs[mod1(loc - 1, end)]; alg=TensorKit.LQpos())
            temp_ACs[loc] = _transpose_front(Qc' * QAc)
        end
        newΨ = InfiniteMPS(Ψ.CR[0], temp_ACs; tol=alg.tolgauge, maxiter=alg.gaugemaxiter)
    end

    recalculate!(envs, newΨ)
    return newΨ, envs
end

function ltr_sweep!(Ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP,
                    envs::Union{Cache,MultipleEnvironments}=environments(Ψ, H))

    # sweep left to right
    for i in 1:(length(Ψ) - 1)
        h_ac = ∂∂AC(i, Ψ, H, envs)
        Ψ.AC[i] = integrate(h_ac, Ψ.AC[i], t, dt, alg.integrator)

        h_c = ∂∂C(i, Ψ, H, envs)
        Ψ.CR[i] = integrate(h_c, Ψ.CR[i], t, -dt, alg.integrator)
    end

    # edge case
    h_ac = ∂∂AC(length(Ψ), Ψ, H, envs)
    Ψ.AC[end] = integrate(h_ac, Ψ.AC[end], t, dt, alg.integrator)

    return Ψ, envs
end

function rtl_sweep!(Ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP,
                    envs::Union{Cache,MultipleEnvironments}=environments(Ψ, H))

    # sweep right to left
    for i in length(Ψ):-1:2
        h_ac = ∂∂AC(i, Ψ, H, envs)
        Ψ.AC[i] = integrate(h_ac, Ψ.AC[i], t, dt, alg.integrator)

        h_c = ∂∂C(i - 1, Ψ, H, envs)
        Ψ.CR[i - 1] = integrate(h_c, Ψ.CR[i - 1], t, -dt, alg.integrator)
    end

    # edge case
    h_ac = ∂∂AC(1, Ψ, H, envs)
    Ψ.AC[1] = integrate(h_ac, Ψ.AC[1], t + dt, dt, alg.integrator)

    return Ψ, envs
end

"""
    TDVP2{A} <: Algorithm

2-site [TDVP](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.070601)
algorithm for time evolution.

# Fields
- `integrator::A`: integrator algorithm (defaults to Lanczos exponentiation)
- `tolgauge::Float64`: tolerance for gauging algorithm
- `gaugemaxiter::Int`: maximum amount of gauging iterations
- `trscheme`: truncation algorithm for [tsvd][TensorKit.tsvd](@ref)
- `finalize::F`: user-supplied function which is applied after each timestep, with
    signature `finalize(t, Ψ, H, envs) -> Ψ, envs`
"""
@kwdef struct TDVP2{A,F} <: Algorithm
    integrator::A = Defaults.alg_expsolve()
    tolgauge::Float64 = Defaults.tolgauge
    gaugemaxiter::Int = Defaults.maxiter
    trscheme = truncerr(1e-3)
    finalize::F = Defaults._finalize
end

function ltr_sweep!(Ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP2,
                    envs=environments(Ψ, H))

    # sweep left to right
    for i in 1:(length(Ψ) - 1)
        ac2 = _transpose_front(Ψ.AC[i]) * _transpose_tail(Ψ.AR[i + 1])
        h_ac2 = ∂∂AC2(i, Ψ, H, envs)
        nac2 = integrate(h_ac2, ac2, t, dt / 2, alg.integrator)

        nal, nc, nar = tsvd!(nac2; trunc=alg.trscheme, alg=TensorKit.SVD())
        Ψ.AC[i] = (nal, complex(nc))
        Ψ.AC[i + 1] = (complex(nc), _transpose_front(nar))

        if i != (length(Ψ) - 1)
            Ψ.AC[i + 1] = integrate(∂∂AC(i + 1, Ψ, H, envs), Ψ.AC[i + 1], t, -dt / 2,
                                    alg.integrator)
        end
    end

    return Ψ, envs
end

function rtl_sweep!(Ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP2,
                    envs=environments(Ψ, H))

    # sweep right to left
    for i in length(Ψ):-1:2
        ac2 = _transpose_front(Ψ.AL[i - 1]) * _transpose_tail(Ψ.AC[i])
        h_ac2 = ∂∂AC2(i - 1, Ψ, H, envs)
        nac2 = integrate(h_ac2, ac2, t + dt / 2, dt / 2, alg.integrator)

        nal, nc, nar = tsvd!(nac2; trunc=alg.trscheme, alg=TensorKit.SVD())
        Ψ.AC[i - 1] = (nal, complex(nc))
        Ψ.AC[i] = (complex(nc), _transpose_front(nar))

        if i != 2
            Ψ.AC[i - 1] = integrate(∂∂AC(i - 1, Ψ, H, envs), Ψ.AC[i - 1], t + dt / 2,
                                    -dt / 2, alg.integrator)
        end
    end

    return Ψ, envs
end
