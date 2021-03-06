"""
Use krylovkit to perform exact diagonalization
"""
function exact_diagonalization(opp::MPOHamiltonian;sector = first(sectors(oneunit(opp.pspaces[1]))),len::Int = opp.period,num::Int = 1,which::Symbol=:SR,alg::KrylovKit.KrylovAlgorithm = Lanczos())
    left = ℂ[typeof(sector)](sector => 1);
    right = oneunit(left);

    middle_site = Int(round(len/2));

    Ot = eltype(opp);

    mpst_type = tensormaptype(spacetype(Ot),2,1,eltype(Ot));
    mpsb_type = tensormaptype(spacetype(Ot),1,1,eltype(Ot));
    CLs = Vector{Union{Missing,mpsb_type}}(missing,len+1);
    ALs = Vector{Union{Missing,mpst_type}}(missing,len);
    ARs = Vector{Union{Missing,mpst_type}}(missing,len);
    ACs = Vector{Union{Missing,mpst_type}}(missing,len);

    for i in 1:middle_site-1
        ALs[i] = isomorphism(storagetype(Ot),left*opp.pspaces[i],fuse(left*opp.pspaces[i]));
        left = _lastspace(ALs[i])';
    end
    for i in len:-1:middle_site+1
        ARs[i] = _permute_front(isomorphism(storagetype(Ot),fuse(opp.pspaces[i]'*right),opp.pspaces[i]'*right));
        right = _firstspace(ARs[i]);
    end
    ACs[middle_site] = TensorMap(rand,ComplexF64,left*opp.pspaces[middle_site],right);
    norm(ACs[middle_site]) == 0 && throw(ArgumentError("invalid sector"));
    normalize!(ACs[middle_site]);

    #construct the largest possible finite mps of that length
    state = FiniteMPS{mpst_type,mpsb_type}(ALs,ARs,ACs,CLs);
    envs = environments(state,opp);

    #optimize the middle site. Because there is no truncation, this single site captures the entire possible hilbert space
    (vals,vecs,convhist) = eigsolve(state.AC[middle_site],num,which,alg) do x
        ac_prime(x,middle_site,state,envs)
    end

    state_vecs = map(vecs) do v
        cs = copy(state);
        cs.AC[middle_site] = v;
        cs
    end

    return vals,state_vecs,convhist
end
