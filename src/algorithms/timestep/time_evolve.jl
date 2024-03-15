"""
    time_evolve(ψ₀, H, t_span, [alg], [envs]; kwargs...)
    time_evolve!(ψ₀, H, t_span, [alg], [envs]; kwargs...)

Time-evolve the initial state `ψ₀` with Hamiltonian `H` over a given time span by stepping
through each of the time points obtained by iterating t_span.

# Arguments
- `ψ₀::AbstractMPS`: initial state
- `H::AbstractMPO`: operator that generates the time evolution (can be time-dependent).
- `t_span::AbstractVector{<:Number}`: time points over which the time evolution is stepped
- `[alg]`: algorithm to use for the time evolution. Defaults to [`TDVP`](@ref).
- `[envs]`: MPS environment manager
"""
function time_evolve end, function time_evolve! end

for (timestep, time_evolve) in zip((:timestep, :timestep!), (:time_evolve, :time_evolve!))
    @eval function $time_evolve(ψ, H, t_span::AbstractVector{<:Number}, alg,
                                envs=environments(ψ, H);
                                verbosity::Int=0)
        log = IterLog(string(nameof(typeof(alg))))
        LoggingExtras.withlevel(; alg.verbosity) do
            @infov 2 loginit!(log, 0, t)
            for iter in 1:(length(t_span) - 1)
                t = t_span[iter]
                dt = t_span[iter + 1] - t

                ψ, envs = $timestep(ψ, H, t, dt, alg, envs)
                ψ, envs = alg.finalize(t, ψ, H, envs)::Tuple{typeof(ψ),typeof(envs)}

                @infov 3 logiter!(log, iter, 0, t)
            end
            @infov 2 logfinish!(log, length(t_span), 0, t_span[end])
        end
        return ψ, envs
    end
end

"""
    timestep(ψ₀, H, t, dt, [alg], [envs]; kwargs...)
    timestep!(ψ₀, H, t, dt, [alg], [envs]; kwargs...)

Time-step the state `ψ₀` with Hamiltonian `H` over a given time step `dt` at time `t`,
solving the Schroedinger equation: ``i ∂ψ/∂t = H ψ``.

# Arguments
- `ψ₀::AbstractMPS`: initial state
- `H::AbstractMPO`: operator that generates the time evolution (can be time-dependent).
- `t::Number`: starting time of time-step
- `dt::Number`: time-step magnitude
- `[alg]`: algorithm to use for the time evolution. Defaults to [`TDVP`](@ref).
- `[envs]`: MPS environment manager
"""
function timestep end, function timestep! end
