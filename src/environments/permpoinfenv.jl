# --- above === below ---
"
    This object manages the periodic mpo environments for an MPSMultiline
"
mutable struct PerMPOInfEnv{H<:MPOMultiline,V,S<:MPSMultiline} <: AbstractInfEnv
    opp :: H

    dependency :: S
    tol :: Float64
    maxiter :: Int

    lw :: PeriodicArray{V,2}
    rw :: PeriodicArray{V,2}

    lock :: ReentrantLock
end

function recalculate!(envs::PerMPOInfEnv,nstate::MPSMultiline)
    sameDspace = reduce((prev,i) -> prev && _lastspace(envs.lw[i...]) == _firstspace(nstate.CR[i...])',
        Iterators.product(1:size(nstate,1),1:size(nstate,2)),init=true);

    init = collect(zip(envs.lw[:,1],envs.rw[:,end]))
    if !sameDspace
        init = gen_init_fps(nstate,envs.opp,nstate)
    end

    (envs.lw,envs.rw) = mixed_fixpoints(nstate,envs.opp,nstate,init,tol = envs.tol, maxiter = envs.maxiter);
    envs.dependency = nstate;

    envs
end

environments(state::InfiniteMPS,opp::InfiniteMPO;kwargs...) = environments(convert(MPSMultiline,state),convert(MPOMultiline,opp);kwargs...);
function environments(state::MPSMultiline,mpo::MPOMultiline;tol = Defaults.tol,maxiter=Defaults.maxiter)
    (lw,rw) = mixed_fixpoints(state,mpo,state;tol = tol, maxiter = maxiter)

    PerMPOInfEnv(mpo,state,tol,maxiter,lw,rw,ReentrantLock())
end

mutable struct MixPerMPOInfEnv{H<:MPOMultiline,V,S<:MPSMultiline} <: AbstractInfEnv
    opp :: H

    above :: S
    dependency :: S

    tol :: Float64
    maxiter :: Int

    lw :: PeriodicArray{V,2}
    rw :: PeriodicArray{V,2}

    lock :: ReentrantLock
end

function recalculate!(envs::MixPerMPOInfEnv,nstate::MPSMultiline)
    sameDspace = reduce((prev,i) -> prev && _firstspace(envs.lw[i...]) == _firstspace(nstate.CR[i...]),
        Iterators.product(1:size(nstate,1),1:size(nstate,2)),init=true);

    init = collect(zip(envs.lw[:,1],envs.rw[:,end]))
    if !sameDspace
        init = gen_init_fps(envs.above,envs.opp,nstate)
    end

    (envs.lw,envs.rw) = mixed_fixpoints(envs.above,envs.opp,nstate,init,tol = envs.tol, maxiter = envs.maxiter);
    envs.dependency = nstate;

    envs
end

function environments(below::InfiniteMPS,toapprox::Tuple{<:InfiniteMPO,<:InfiniteMPS};kwargs...)
    (opp,above) = toapprox
    environments(convert(MPSMultiline,below),(convert(MPOMultiline,opp),convert(MPSMultiline,above));kwargs...);
end
function environments(below::MPSMultiline,toapprox::Tuple{<:MPOMultiline,<:MPSMultiline};tol = Defaults.tol,maxiter=Defaults.maxiter)
    (mpo,above) = toapprox;
    (lw,rw) = mixed_fixpoints(above,mpo,below;tol = tol, maxiter = maxiter)

    MixPerMPOInfEnv(mpo,above,below,tol,maxiter,lw,rw,ReentrantLock())
end

# --- utility functions ---

function gen_init_fps(above::MPSMultiline,mpo::MPOMultiline,below::MPSMultiline)
    T = eltype(above)

    map(1:size(mpo,1)) do cr
        L0::T = TensorMap(rand,eltype(T),space(below.AL[cr,1],1)*space(mpo[cr,1],1)',space(above.AL[cr,1],1))
        R0::T = TensorMap(rand,eltype(T),space(above.AR[cr,1],1)*space(mpo[cr,1],1),space(below.AR[cr,1],1))
        (L0,R0)
    end
end

function mixed_fixpoints(above::MPSMultiline,mpo::MPOMultiline,below::MPSMultiline,init = gen_init_fps(above,mpo,below);tol = Defaults.tol, maxiter = Defaults.maxiter)
    T = eltype(above);

    #sanity check
    (numrows,numcols) = size(above)
    @assert size(above) == size(mpo)
    @assert size(below) == size(mpo);

    lefties = PeriodicArray{T,2}(undef,numrows,numcols);
    righties = PeriodicArray{T,2}(undef,numrows,numcols);

    @sync for cr = 1:numrows
        @Threads.spawn begin
            (L0,R0) = $init[cr]

            (_,Ls,convhist) = eigsolve(x-> transfer_left(x,$mpo[cr,:],$above.AL[cr,:],$below.AL[cr+1,:]),L0,1,:LM,Arnoldi(tol = tol,maxiter=maxiter))
            convhist.converged < 1 && @info "left eigenvalue failed to converge $(convhist.normres)"
            (_,Rs,convhist) = eigsolve(x-> transfer_right(x,$mpo[cr,:],$above.AR[cr,:],$below.AR[cr+1,:]),R0,1,:LM,Arnoldi(tol = tol,maxiter=maxiter))
            convhist.converged < 1 && @info "right eigenvalue failed to converge $(convhist.normres)"


            $lefties[cr,1] = Ls[1]
            for loc in 2:numcols
                $lefties[cr,loc] = transfer_left($lefties[cr,loc-1],$mpo[cr,loc-1],$above.AL[cr,loc-1],$below.AL[cr+1,loc-1])
            end

            #@plansor renormfact[] := Ls[1][1 2;3]*above.CR[cr,0][3;4]*Rs[1][4 2;5]*conj(below.CR[cr+1,0][1;5])
            @plansor t_renormfact[-1;-2] := Ls[1][-1 2;3]*above.CR[cr,0][3;4]*Rs[1][4 2;5]*conj(below.CR[cr+1,0][-2;5])
            renormfact = tr(t_renormfact);

            $righties[cr,end] = Rs[1]/sqrt(renormfact);
            $lefties[cr,1] /=sqrt(renormfact);

            for loc in numcols-1:-1:1
                $righties[cr,loc] = transfer_right($righties[cr,loc+1],$mpo[cr,loc+1],$above.AR[cr,loc+1],$below.AR[cr+1,loc+1])

                #@plansor renormfact[] := lefties[cr,loc+1][1 2;3]*above.CR[cr,loc][3;4]*righties[cr,loc][4 2;5]*conj(below.CR[cr+1,loc][1;5])
                @plansor t_renormfact[-1;-2] := lefties[cr,loc+1][-1 2;3]*above.CR[cr,loc][3;4]*righties[cr,loc][4 2;5]*conj(below.CR[cr+1,loc][-2;5])
                renormfact = tr(t_renormfact);

                $righties[cr,loc]/=sqrt(renormfact)
                $lefties[cr,loc+1]/=sqrt(renormfact)
            end
        end
    end

    return (lefties,righties)
end
