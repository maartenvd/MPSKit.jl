# This file contains the definition of the IterativeSolver type and the solve! function.
# Attempts to remove as much of the boilerplate code as possible from the iterative solvers.

mutable struct IterativeSolver{A,B}
    alg::A
    state::B
end

# Iteration
# ---------

function Base.iterate(alg::IterativeSolver, state=alg.state)
    if isfinished(alg)
        logfinish(alg)
        return nothing
    elseif iscancelled(alg)
        logcancel(alg)
        return nothing
    end

    item = iterate!(alg)
    logiter(alg)

    return item, alg.state
end

function solve!(args...)
    return LoggingExtras.withlevel(; args[end].verbosity) do
        iterator = initialize!(args...)
        for _ in iterator
        end
        return finalize!(iterator)
    end
end

# Logging
# -------
# Defaults to very simple messages

alg_id(alg::IterativeSolver) = objectid(alg)
alg_group(alg::IterativeSolver) = nameof(typeof(alg.alg))
