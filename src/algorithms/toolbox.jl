"calculates the entropy of a given state"
entropy(state::InfiniteMPS) = [tr(c*log(c)) for c in state.CR]

"
given a thermal state, you can map it to an mps by fusing the physical legs together
to prepare a gibbs ensemble, you need to evolve this state with H working on both legs
here we return the 'superhamiltonian' (H*id,id*H)
"
function splitham(ham::MPOHamiltonian)
    fusers = [isomorphism(fuse(p*p'),p*p') for p in ham.pspaces]

    idham = Array{Union{Missing,typeof(ham[1,1,1])},3}(missing,ham.period,ham.odim,ham.odim)
    hamid = Array{Union{Missing,typeof(ham[1,1,1])},3}(missing,ham.period,ham.odim,ham.odim)

    for i in 1:ham.period
        for (k,l) in keys(ham,i)
            idt = isomorphism(ham.pspaces[i],ham.pspaces[i])

            @tensor hamid[i,k,l][-1 -2; -3 -4] :=   ham[i,k,l][-1,12,-3,15]*
                                                    idt[16,13]*
                                                    fusers[i][-2,12,13]*
                                                    conj(fusers[i][-4,15,16])

            @tensor idham[i,k,l][-1 -2; -3 -4] :=   ham[i,k,l][-1,13,-3,16]*
                                                    idt[12,15]*
                                                    fusers[i][-2,12,16]*
                                                    conj(fusers[i][-4,15,13])
        end
    end

    return MPOHamiltonian(hamid),MPOHamiltonian(idham)
end

infinite_temperature(ham::MPOHamiltonian) = [permute(isomorphism(Matrix{eltype(ham[1,1,1])},oneunit(sp)*sp,oneunit(sp)*sp),(1,2,4),(3,)) for sp in ham.pspaces]

"calculates the galerkin error"
calc_galerkin(state::Union{InfiniteMPS,FiniteMPS,MPSComoving},loc,envs)::Float64 = norm(leftnull(state.AC[loc])'*ac_prime(state.AC[loc], loc,state,envs))
calc_galerkin(state::Union{InfiniteMPS,FiniteMPS,MPSComoving}, envs)::Float64 = maximum([calc_galerkin(state,loc,envs) for loc in 1:length(state)])
calc_galerkin(state::MPSMultiline, envs::PerMPOInfEnv)::Float64 = maximum([norm(leftnull(state.AC[row+1,col])'*ac_prime(state.AC[row,col], row,col,state,envs)) for (row,col) in Iterators.product(1:size(state,1),1:size(state,2))][:])
calc_galerkin(state::MPSMultiline, envs::MixPerMPOInfEnv)::Float64 = maximum([norm(leftnull(state.AC[row+1,col])'*ac_prime(envs.above.AC[row,col], row,col,state,envs)) for (row,col) in Iterators.product(1:size(state,1),1:size(state,2))][:])

"
Calculates the (partial) transfer spectrum
"
function transfer_spectrum(above::InfiniteMPS;below=above,tol=Defaults.tol,num_vals = 20,sector=first(sectors(oneunit(virtualspace(above,1)))))
    init = TensorMap(rand, eltype(eltype(above)), virtualspace(below,0)*ℂ[sector => 1],virtualspace(above,0))

    num_vals = min(dim(virtualspace(above,0)*virtualspace(below,0)),num_vals); # we can ask at most this many values

    eigenvals, eigenvecs,convhist = eigsolve(x->transfer_left(x, above.AL, below.AL) , init, num_vals, :LM, tol=tol)
    convhist.converged < num_vals && @warn "correlation length failed to converge $(convhist.normres)"

    return eigenvals
end

"
Returns the (full) entanglement spectrum at site I
"
function entanglement_spectrum(st::Union{InfiniteMPS,FiniteMPS,MPSComoving},site::Int=0)
    @assert site<=length(st)

    (U,S,V) = tsvd(st.CR[site]);
    diag(convert(Array,S))
end

function correlation_length(above::InfiniteMPS; tol_angle=0.1,below=above,kwargs...)
    #get the transfer spectrum
    spectrum = transfer_spectrum(above;below=above,kwargs...);

    correlation_length(spectrum,above;tol_angle=tol_angle,below=below,kwargs...)
end

function correlation_length(spectrum,above::InfiniteMPS; tol_angle=0.1,below=above,kwargs...)
    #we also define a correlation length between different states
    (above === below) && (spectrum = spectrum[2:end])

    best_angle = mod1(angle(spectrum[1]), 2*pi)
    ind_at_angle = findall(x->x<tol_angle || abs(x-2*pi)<tol_angle, mod1.(angle.(spectrum).-best_angle, 2*pi))
    spectrum_at_angle = spectrum[ind_at_angle]

    lambdas = -log.(abs.(spectrum_at_angle));

    corlength = 1/first(lambdas);

    gap = Inf;
    if length(lambdas) > 2
        gap = lambdas[2]-lambdas[1]
    end

    return corlength, gap, best_angle
end


function variance(state::InfiniteMPS,ham::MPOHamiltonian,envs = environments(state,ham))
    rescaled_ham = ham-expectation_value(state,ham,envs);
    real(sum(expectation_value(state,rescaled_ham*rescaled_ham)))
end

function variance(state::FiniteMPS,ham::MPOHamiltonian,envs = environments(state,ham))
    ham2 = ham*ham;
    real(sum(expectation_value(state,ham2)) - sum(expectation_value(state,ham,envs))^2)
end

function variance(state::MPSComoving,ham::MPOHamiltonian,envs = environments(state,ham))
    #tricky to define
    (ham2,nenvs) = squaredenvs(state,ham,envs);
    real(expectation_value(state,ham2,nenvs)[2] - expectation_value(state,ham,envs)[2]^2)
end

variance(state::FiniteQP,ham::MPOHamiltonian,args...) = variance(convert(FiniteMPS,state),ham);

function variance(state::InfiniteQP,ham::MPOHamiltonian,envs=environments(state,ham))
    # I remember there being an issue here @gertian?
    state.trivial || throw(ArgumentError("variance of domain wall excitations is not implemented"));

    rescaled_ham = ham - expectation_value(state.left_gs,ham);

    #I don't remember where the formula came from
    E_ex = dot(state,effective_excitation_hamiltonian(ham,state,envs));
    E_f = expectation_value(state.left_gs,rescaled_ham,0);

    ham2 = rescaled_ham*rescaled_ham

    real(dot(state,effective_excitation_hamiltonian(ham2,state))-2*(E_f+E_ex)*E_ex+E_ex^2)
end

"""
You can impose periodic boundary conditions on an mpo-hamiltonian (for a given size)
That creates a new mpo-hamiltonian with larger bond dimension
The interaction never wraps around multiple times
"""
function periodic_boundary_conditions(ham::MPOHamiltonian{S,T,E},len = ham.period) where {S,T,E}
    sanitycheck(ham) || throw(ArgumentError("invalid ham"))
    mod(len,ham.period) == 0 || throw(ArgumentError("$(len) is not a multiple of unitcell"))

    fusers = PeriodicArray(map(1:len) do loc
        map(Iterators.product(ham.domspaces[len+1,:],ham.domspaces[loc,:],ham.domspaces[loc,:])) do (v1,v2,v3)
            isomorphism(fuse(v1'*v2*v3),v1'*v2*v3)
        end
    end)

    #a -> what virtual space did I "lend" in the beginning?
    #b -> what progress have I made in the lower layer?
    #c -> what progress have I made in the upper layer?
    χ = ham.odim;
    χ´ = Int((χ-1)*χ*(χ+1)/2+1);

    function indmap(a,b,c)
        Int((χ-a)*(χ-a+1)/2+(c-1)*χ*(χ+1)/2+(b-a)+1)
    end


    #do the bulk
    bulk = PeriodicArray(convert(Array{Union{T,E},3},fill(zero(E),ham.period,χ´,χ´)));



    for loc in 1:ham.period,
        (j,k) in keys(ham,loc)

        #apply (j,k) above
        l = ham.odim
        for i in 2:ham.odim

            k <= i && i<=l || continue

            f1 = fusers[loc][i,l,j]
            f2 = fusers[loc+1][i,l,k]
            @tensor bulk[loc,indmap(i,l,j),indmap(i,l,k)][-1 -2;-3 -4]:=ham[loc,j,k][1,-2,2,-4]*f1[-1,3,4,1]*conj(f2[-3,3,4,2])
        end


        #apply (j,k) below
        i = 1
        for l in 2:(ham.odim-1)

            l > 1 && l >= i && l<=j || continue

            f1 = fusers[loc][l,j,i];
            f2 = fusers[loc+1][l,k,i];

            @tensor bulk[loc,indmap(l,j,i),indmap(l,k,i)][-1 -2;-3 -4]:=ham[loc,j,k][1,-2,2,-4]*f1[-1,3,1,4]*conj(f2[-3,3,2,4])
        end
    end


    # make the starter
    starter = convert(Array{Union{T,E},2},fill(zero(E),χ´,χ´));
    for (j,k) in keys(ham,1)

        #apply (j,k) above
        if j == 1
            f1 = fusers[1][end,end,1];
            f2 = fusers[1][end,end,k];

            @tensor starter[1,indmap(ham.odim,ham.odim,k)][-1 -2;-3 -4]:=
            ham[1,j,k][1,-2,2,-4]*f1[-1,3,4,1]*conj(f2[-3,3,4,2])
        end

        #apply (j,k) below
        if j > 1 && j < ham.odim

            f1 = fusers[1][j,j,1];
            f2 = fusers[1][j,k,1];

            @tensor starter[1,indmap(j,k,1)][-1 -2;-3 -4]:=ham[1,j,k][1,-2,2,-4]*conj(f2[-3,1,2,-1])
        end
    end
    starter[1,1] = one(E);
    starter[end,end] = one(E);

    # make the ender
    ender = convert(Array{Union{T,E},2},fill(zero(E),χ´,χ´));
    for (j,k) in keys(ham,ham.period)

        if k > 1
            f1 = fusers[end][k,ham.odim,j]
            @tensor ender[indmap(k,ham.odim,j),end][-1 -2;-3 -4]:=ham[ham.period,j,k][1,-2,2,-4]*f1[-1,2,-3,1]
        end
    end
    ender[1,1] = one(E);
    ender[end,end] = one(E);

    # fill in the entire ham
    nos = convert(Array{Union{T,E},3},fill(zero(E),len,χ´,χ´))
    nos[1,:,:] = starter[:,:];
    nos[end,:,:] = ender[:,:];

    for i in 2:len-1
        nos[i,:,:] = bulk[i,:,:];
    end



    return MPOHamiltonian(nos)
end

#impose periodic boundary conditions on a normal mpo
function periodic_boundary_conditions(mpo::InfiniteMPO{O},len = length(mpo)) where O
    mod(len,length(mpo)) == 0 || throw(ArgumentError("len not a multiple of unitcell"))

    output = PeriodicArray{O,1}(undef,len);


    sp = space(mpo[1],1)';
    utleg = Tensor(ones,oneunit(sp));

    #do the bulk
    for j in 2:len-1
        f1 = isomorphism(fuse(sp*space(mpo[j],1)),sp*space(mpo[j],1))
        f2 = isomorphism(fuse(sp*space(mpo[j],3)'),sp*space(mpo[j],3)')

        @tensor output[j][-1 -2;-3 -4] := mpo[j][1,-2,2,-4]*f1[-1,3,1]*conj(f2[-3,3,2])
    end

    #do the left
    f2 = isomorphism(fuse(sp*space(mpo[1],3)'),sp*space(mpo[1],3)')
    @tensor output[1][-1 -2;-3 -4] := mpo[1][1,-2,2,-4]*conj(f2[-3,1,2])*utleg[-1]

    #do the right
    f2 = isomorphism(fuse(sp*space(mpo[len],1)),sp*space(mpo[len],1))
    @tensor output[end][-1 -2;-3 -4] := mpo[len][1,-2,2,-4]*f2[-1,2,1]*conj(utleg[-3])

    InfiniteMPO(output)
end
