module MPSKit

using TensorKit, KrylovKit, OptimKit, FastClosures
using Base.Threads, FLoops, Transducers, FoldsThreads
using Base.Iterators
using RecipesBase
using VectorInterface
using Accessors

using LinearAlgebra: diag, Diagonal
using LinearAlgebra: LinearAlgebra
using Base: @kwdef
using LoggingExtras

# bells and whistles for mpses
export InfiniteMPS, FiniteMPS, WindowMPS, MPSMultiline
export PeriodicArray, Window
export MPSTensor
export QP, LeftGaugedQP, RightGaugedQP
export leftorth,
       rightorth, leftorth!, rightorth!, invalidate!, uniform_leftorth, uniform_rightorth
export r_LL, l_LL, r_RR, l_RR, r_RL, r_LR, l_RL, l_LR # should be properties

# useful utility functions?
export add_util_leg, max_Ds, recalculate!
export left_virtualspace, right_virtualspace, physicalspace
export entanglementplot, transferplot

# hamiltonian things
export Cache
export SparseMPO, MPOHamiltonian, DenseMPO, MPOMultiline, FiniteMPO
export UntimedOperator, TimedOperator, MultipliedOperator, LazySum

export ∂C, ∂AC, ∂AC2, environments, expectation_value, effective_excitation_hamiltonian
export leftenv, rightenv

# algos
export find_groundstate!, find_groundstate, leading_boundary
export VUMPS, VOMPS, DMRG, DMRG2, IDMRG1, IDMRG2, GradientGrassmann
export excitations, FiniteExcited, QuasiparticleAnsatz, ChepigaAnsatz, ChepigaAnsatz2
export marek_gap, correlation_length, correlator
export time_evolve, time_evolve!, timestep!, timestep
export TDVP, TDVP2, WindowTDVP, make_time_mpo, WI, WII, TaylorCluster
export splitham, infinite_temperature, entanglement_spectrum, transfer_spectrum, variance
export changebonds!, changebonds, VUMPSSvdCut, OptimalExpand, SvdCut, UnionTrunc, RandExpand
export entropy
export propagator, NaiveInvert, Jeckelmann, DynamicalDMRG
export fidelity_susceptibility
export approximate!, approximate
export periodic_boundary_conditions
export exact_diagonalization

# transfer matrix
export TransferMatrix
export transfer_left, transfer_right

@deprecate virtualspace left_virtualspace # there is a possible ambiguity when C isn't square, necessitating specifying left or right virtualspace
@deprecate params(args...) environments(args...)
@deprecate InfiniteMPO(args...) DenseMPO(args...)

# Abstract type defs
abstract type Algorithm end
abstract type Cache end # cache "manages" environments

# submodules
include("utility/dynamictols.jl")
using .DynamicTols

include("utility/defaults.jl")
using .Defaults: VERBOSE_NONE, VERBOSE_WARN, VERBOSE_CONV, VERBOSE_ITER, VERBOSE_ALL
include("utility/logging.jl")
using .IterativeLoggers
include("utility/iterativesolvers.jl")

include("utility/periodicarray.jl")
include("utility/multiline.jl")
include("utility/utility.jl") # random utility functions
include("utility/plotting.jl")
include("utility/linearcombination.jl")
include("utility/timedependence.jl")

# maybe we should introduce an abstract state type
include("states/window.jl")
include("states/abstractmps.jl")
include("states/infinitemps.jl")
include("states/mpsmultiline.jl")
include("states/finitemps.jl")
include("states/windowmps.jl")
include("states/orthoview.jl")
include("states/quasiparticle_state.jl")
include("states/ortho.jl")

include("operators/densempo.jl")
include("operators/sparsempo/sparseslice.jl")
include("operators/sparsempo/sparsempo.jl")
include("operators/mpohamiltonian.jl") # the mpohamiltonian objects
include("operators/mpomultiline.jl")
include("operators/projection.jl")
include("operators/multipliedoperator.jl")
include("operators/lazysum.jl")

include("transfermatrix/transfermatrix.jl")
include("transfermatrix/transfer.jl")

include("environments/finenv.jl")
include("environments/abstractinfenv.jl")
include("environments/permpoinfenv.jl")
include("environments/mpohaminfenv.jl")
include("environments/windowenv.jl")
include("environments/qpenv.jl")
include("environments/multipleenv.jl")
include("environments/idmrgenv.jl")
include("environments/lazylincocache.jl")

include("algorithms/fixedpoint.jl")
include("algorithms/derivatives.jl")
include("algorithms/expval.jl")
include("algorithms/toolbox.jl")
include("algorithms/grassmann.jl")
include("algorithms/correlators.jl")

include("algorithms/changebonds/changebonds.jl")
include("algorithms/changebonds/optimalexpand.jl")
include("algorithms/changebonds/vumpssvd.jl")
include("algorithms/changebonds/svdcut.jl")
include("algorithms/changebonds/randexpand.jl")

include("algorithms/timestep/tdvp.jl")
include("algorithms/timestep/timeevmpo.jl")
include("algorithms/timestep/integrators.jl")
include("algorithms/timestep/time_evolve.jl")
include("algorithms/timestep/windowtdvp.jl")
include("algorithms/timestep/timestep.jl")

include("algorithms/groundstate/vumps.jl")
include("algorithms/groundstate/idmrg.jl")
include("algorithms/groundstate/dmrg.jl")
include("algorithms/groundstate/gradient_grassmann.jl")
include("algorithms/groundstate/find_groundstate.jl")

include("algorithms/propagator/corvector.jl")

include("algorithms/excitation/excitations.jl")
include("algorithms/excitation/quasiparticleexcitation.jl")
include("algorithms/excitation/dmrgexcitation.jl")
include("algorithms/excitation/chepigaansatz.jl")
include("algorithms/excitation/exci_transfer_system.jl")

include("algorithms/statmech/vumps.jl")
include("algorithms/statmech/vomps.jl")
include("algorithms/statmech/gradient_grassmann.jl")

include("algorithms/fidelity_susceptibility.jl")

include("algorithms/approximate/approximate.jl")
include("algorithms/approximate/vomps.jl")
include("algorithms/approximate/fvomps.jl")
include("algorithms/approximate/idmrg.jl")

include("algorithms/ED.jl")

include("algorithms/unionalg.jl")

# include("precompile.jl")
# _precompile_()

end
