module MPSKit
    using TensorKit,KrylovKit,Parameters, Base.Threads,OptimKit, FastClosures

    import TensorOperations as TO;
    using LinearAlgebra:diag,Diagonal;
    import LinearAlgebra

    #bells and whistles for mpses
    export InfiniteMPS,FiniteMPS,MPSComoving,PeriodicArray,MPSMultiline
    export transfer_left,transfer_right
    export leftorth,rightorth,leftorth!,rightorth!,poison!,uniform_leftorth,uniform_rightorth
    export r_LL,l_LL,r_RR,l_RR,r_RL,r_LR,l_RL,l_LR #should be properties

    #useful utility functions?
    export spinmatrices,add_util_leg,full,nonsym_spintensors,nonsym_bosonictensors
    export max_Ds,virtualspace, reorth!, recalculate!

    #hamiltonian things
    export Hamiltonian,Operator,Cache
    export MPOHamiltonian,InfiniteMPO,MPOMultiline
    export ac_prime,c_prime,environments,ac2_prime,expectation_value,effective_excitation_hamiltonian
    export leftenv,rightenv

    #algos
    export find_groundstate!, find_groundstate, Vumps, Dmrg, Dmrg2, GradDesc, Idmrg1, Idmrg2, GradientGrassmann
    export leading_boundary
    export correlation_length,excitations,FiniteExcited,QuasiparticleAnsatz
    export timestep!,timestep,Tdvp,Tdvp2,make_time_mpo,WI,WII
    export splitham,mpo2mps,mps2mpo,infinite_temperature, entanglement_spectrum, transfer_spectrum, variance
    export changebonds!,changebonds,VumpsSvdCut,OptimalExpand,SvdCut,UnionTrunc,RandExpand
    export entropy
    export dynamicaldmrg
    export fidelity_susceptibility
    export approximate!,approximate, Vomps
    export periodic_boundary_conditions
    export exact_diagonalization

    @deprecate params(args...) environments(args...)

    #default settings
    module Defaults
        const eltype = ComplexF64
        const maxiter = 100
        const tolgauge = 1e-14
        const tol = 1e-12
        const verbose = true
        _finalize(iter,state,opp,envs) = (state,envs);
    end

    include("utility/periodicarray.jl")
    include("utility/utility.jl") #random utility functions

    #maybe we should introduce an abstract state type
    include("states/abstractmps.jl")
    include("states/transfer.jl") # mps transfer matrices
    include("states/infinitemps.jl")
    include("states/multiline.jl")
    include("states/finitemps.jl")
    include("states/comoving.jl")
    include("states/orthoview.jl")
    include("states/quasiparticle_state.jl")
    include("states/ortho.jl")

    abstract type Operator end
    abstract type Hamiltonian <: Operator end

    include("operators/mpohamiltonian/mpohamiltonian.jl") #the mpohamiltonian objects
    include("operators/umpo.jl")

    abstract type Cache end #cache "manages" environments

    include("environments/FinEnv.jl")
    include("environments/abstractinfenv.jl")
    include("environments/permpoinfenv.jl")
    include("environments/mpohaminfenv.jl")
    include("environments/overlapenv.jl")
    include("environments/qpenv.jl")
    include("environments/idmrgenv.jl")

    abstract type Algorithm end

    include("algorithms/derivatives.jl")
    include("algorithms/expval.jl")
    include("algorithms/toolbox.jl")
    include("algorithms/grassmann.jl")

    include("algorithms/changebonds/optimalexpand.jl")
    include("algorithms/changebonds/vumpssvd.jl")
    include("algorithms/changebonds/svdcut.jl")
    include("algorithms/changebonds/randexpand.jl")

    include("algorithms/timestep/tdvp.jl")
    include("algorithms/timestep/timeevmpo.jl")

    include("algorithms/groundstate/vumps.jl")
    include("algorithms/groundstate/idmrg.jl")
    include("algorithms/groundstate/dmrg.jl")
    include("algorithms/groundstate/gradient_grassmann.jl")

    include("algorithms/propagator/corvector.jl")

    include("algorithms/excitation/quasiparticleexcitation.jl")
    include("algorithms/excitation/dmrgexcitation.jl")

    include("algorithms/statmech/vumps.jl")
    include("algorithms/statmech/gradient_grassmann.jl")

    include("algorithms/fidelity_susceptibility.jl")

    include("algorithms/approximate/vomps.jl")
    include("algorithms/approximate/fvomps.jl")

    include("algorithms/ED.jl")

    include("algorithms/unionalg.jl")

    Base.@deprecate PeriodicMPO InfiniteMPO
end
