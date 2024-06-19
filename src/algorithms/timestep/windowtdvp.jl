"""
    WindowTDVP{A} <: Algorithm

[Mixed TDVP](https://arxiv.org/abs/2007.15035) algorithm for time evolution.

# Fields
- `left::A`: algorithm to do the timestep of the infinite part of the WindowMPS. 
- `middle::B`: algorithm to do the timestep of the finite part of the WindowMPS. This can be `TDVP2` to expand the bonddimension.
- `right::C`: algorithm to do the timestep of the right part of the WindowMPS. By default the same as left
- `finalize::F`: user-supplied function which is applied after each timestep, with
    signature `finalize(t, Ψ, H, envs) -> Ψ, envs`. Can be used to enlarge the bond dimension of the infinite part.
"""
@kwdef struct WindowTDVP{A,B,C,F} <: Algorithm
    left::A = TDVP()
    middle::B = TDVP()
    right::C = left
    finalize::F = Defaults._finalize
end

function timestep!(Ψ::WindowMPS{A,B,VL,VR}, H::Union{Window,LazySum{<:Window}}, t::Number,
                   dt::Number, alg::WindowTDVP, env=environments(Ψ, H)) where {A,B,VL,VR}

    #first evolve left state
    if VL === WINDOW_VARIABLE
        nleft, _ = timestep(Ψ.left, H.left, t, dt, alg.left, env.left; leftorthflag=true) #env gets updated in place
        Ψ = WindowMPS(nleft, Ψ.middle, Ψ.right)
    end

    Ψ, env = ltr_sweep!(Ψ, H.middle, t, dt / 2, alg.middle, env)

    #then evolve right state
    if VR === WINDOW_VARIABLE
        nright, _ = timestep(Ψ.right, H.right, t + dt, dt, alg.right, env.right;
                             leftorthflag=false) # env gets updated in place
        Ψ = WindowMPS(Ψ.left, Ψ.middle, nright)
    end

    Ψ, env = rtl_sweep!(Ψ, H.middle, t + dt / 2, dt / 2, alg.middle, env)

    return Ψ, env
end