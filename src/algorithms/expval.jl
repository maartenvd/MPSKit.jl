"""
    expectation_value(ψ, O, [location], [environments])

Compute the expectation value of an operator `O` on a state `ψ`. If `location` is given, the
operator is applied at that location. If `environments` is given, the expectation value
might be computed more efficiently by re-using previously calculated environments.

!!! note
    For `MPOHamiltonian`, the expectation value is not uniquely defined, as it is unclear to
    what site a given term belongs. For this reason, the returned value is half the
    expectation value of all terms that start and end on the site.

# Arguments
* `ψ::AbstractMPS` : the state on which to compute the expectation value
* `O` : the operator to compute the expectation value of. This can either be an `AbstractMPO`, a single `AbstractTensorMap` or an array of `AbstractTensorMap`s.
* `location::Union{Int,AbstractRange{Int}}` : the location at which to apply the operator. Only applicable for operators that act on a subset of all sites.
* `environments::Cache` : the environments to use for the calculation. If not given, they will be calculated.
"""
function expectation_value end

# Local operators
# ---------------
function expectation_value(ψ::Union{InfiniteMPS,WindowMPS,FiniteMPS},
                           O::AbstractTensorMap{S,1,1}) where {S}
    return expectation_value(ψ, fill(O, length(ψ)))
end
function expectation_value(ψ::Union{InfiniteMPS,WindowMPS,FiniteMPS},
                           opps::AbstractArray{<:AbstractTensorMap{S,1,1}}) where {S}
    return map(zip(ψ.AC, opps)) do (ac, opp)
        return tr(ac' * transpose(opp * transpose(ac,
                                                  ((TensorKit.allind(ac)[2:(end - 1)]),
                                                   (1, TensorKit.numind(ac)))),
                                  ((TensorKit.numind(ac) - 1,
                                    TensorKit.allind(ac)[1:(end - 2)]...),
                                   (TensorKit.numind(ac),))))
    end
end

# Multi-site operators
# --------------------
# TODO: replace Vector{MPOTensor} with FiniteMPO
function expectation_value(ψ::Union{FiniteMPS{T},WindowMPS{T},InfiniteMPS{T}},
                           O::AbstractTensorMap{S,N,N}, at::Int) where {S,N,T<:MPSTensor{S}}
    return expectation_value(ψ, decompose_localmpo(add_util_leg(O)), at)
end
function expectation_value(ψ::Union{FiniteMPS{T},WindowMPS{T},InfiniteMPS{T}},
                           O::AbstractArray{<:MPOTensor{S}},
                           at::Int) where {S,T<:MPSTensor{S}}
    firstspace = _firstspace(first(O))
    (firstspace == oneunit(firstspace) && _lastspace(last(O)) == firstspace') ||
        throw(ArgumentError("localmpo should start and end in a trivial leg, not with $(firstspace)"))

    ut = fill_data!(similar(O[1], firstspace), one)
    @plansor v[-1 -2; -3] := isomorphism(storagetype(T), left_virtualspace(ψ, at - 1),
                                         left_virtualspace(ψ, at - 1))[-1; -3] *
                             conj(ut[-2])
    tmp = v *
          TransferMatrix(ψ.AL[at:(at + length(O) - 1)], O, ψ.AL[at:(at + length(O) - 1)])
    return @plansor tmp[1 2; 3] * ut[2] * ψ.CR[at + length(O) - 1][3; 4] *
                    conj(ψ.CR[at + length(O) - 1][1; 4])
end

# MPOHamiltonian
# --------------
# TODO: add section in documentation to explain convention
expectation_value(ψ, envs::Cache) = expectation_value(ψ, envs.opp, envs)
function expectation_value(ψ, H::Union{MPOHamiltonian,Window,LazySum})
    return expectation_value(ψ, H, environments(ψ, H))
end

function expectation_value(ψ::AbstractFiniteMPS, H::MPOHamiltonian, envs::FinEnv)
    ens = zeros(scalartype(ψ), length(ψ))
    for i in 1:length(ψ), (j, k) in keys(H[i])
        !((j == 1 && k != 1) || (k == H.odim && j != H.odim)) && continue

        cur = @plansor leftenv(envs, i, ψ)[j][1 2; 3] * ψ.AC[i][3 7; 5] *
                       rightenv(envs, i, ψ)[k][5 8; 6] * conj(ψ.AC[i][1 4; 6]) *
                       H[i][j, k][2 4; 7 8]
        if !(j == 1 && k == H.odim)
            cur /= 2
        end

        ens[i] += cur
    end

    n = norm(ψ.AC[end])^2
    return ens ./ n
end

function expectation_value(st::InfiniteMPS, H::MPOHamiltonian,
                           prevca::Union{MPOHamInfEnv,IDMRGEnv})
    #calculate energy density
    len = length(st)
    ens = PeriodicArray(zeros(scalartype(st.AR[1]), len))
    for i in 1:len
        util = fill_data!(similar(st.AL[1], space(prevca.lw[H.odim, i + 1], 2)), one)
        for j in (H.odim):-1:1
            apl = leftenv(prevca, i, st)[j] *
                  TransferMatrix(st.AL[i], H[i][j, H.odim], st.AL[i])
            ens[i] += @plansor apl[1 2; 3] * r_LL(st, i)[3; 1] * conj(util[2])
        end
    end
    return ens
end

#the mpo hamiltonian over n sites has energy f+n*edens, which is what we calculate here. f can then be found as this - n*edens
function expectation_value(st::InfiniteMPS, prevca::MPOHamInfEnv,
                           range::Union{UnitRange{Int},Int})
    return expectation_value(st, prevca.opp, range, prevca)
end
function expectation_value(st::InfiniteMPS, H::MPOHamiltonian, range::Int,
                           prevca=environments(st, H))
    return expectation_value(st, H, 1:range, prevca)
end
function expectation_value(st::InfiniteMPS, H::MPOHamiltonian, range::UnitRange{Int},
                           prevca=environments(st, H))
    start = map(leftenv(prevca, range.start, st)) do y
        @plansor x[-1 -2; -3] := y[1 -2; 3] * st.CR[range.start - 1][3; -3] *
                                 conj(st.CR[range.start - 1][1; -1])
    end

    for i in range
        start = start * TransferMatrix(st.AR[i], H[i], st.AR[i])
    end

    tot = 0.0 + 0im
    for i in 1:(H.odim)
        tot += @plansor start[i][1 2; 3] * rightenv(prevca, range.stop, st)[i][3 2; 1]
    end

    return tot
end

function expectation_value(ψ::WindowMPS, H::MPOHamiltonian, range::UnitRange{Int},
                           envs::FinEnv=environments(ψ, H))
    if range == 1:length(ψ)
        tot = 0.0 + 0im
        for i in 1:(H.odim), j in 1:(H.odim)
            tot += @plansor leftenv(envs, length(ψ), ψ)[i][1 2; 3] * ψ.AC[end][3 4; 5] *
                            rightenv(envs, length(ψ), ψ)[j][5 6; 7] *
                            H[length(ψ)][i, j][2 8; 4 6] * conj(ψ.AC[end][1 8; 7])
        end

        return tot / (norm(ψ.AC[end])^2)
    else
        # probably doesn't make sense for any other range
        # does it even make sense for unequal left and right states?
        throw(ArgumentError("range $range not supported"))
    end
end

function expectation_value(ψ::WindowMPS, H::Window, range::UnitRange{Int},
                           envs::WindowEnv=environments(ψ, H))
    return expectation_value(ψ, H.middle, range, finenv(envs, ψ))
end

# Transfer matrices
# -----------------
function expectation_value(ψ::InfiniteMPS, mpo::DenseMPO)
    return expectation_value(convert(MPSMultiline, ψ), convert(MPOMultiline, mpo))
end
function expectation_value(ψ::MPSMultiline, mpo::MPOMultiline)
    return expectation_value(ψ, environments(ψ, mpo))
end
function expectation_value(ψ::InfiniteMPS, ca::PerMPOInfEnv)
    return expectation_value(convert(MPSMultiline, ψ), ca)
end
function expectation_value(ψ::MPSMultiline, O::MPOMultiline, ca::PerMPOInfEnv)
    retval = PeriodicMatrix{scalartype(ψ)}(undef, size(ψ, 1), size(ψ, 2))
    for (i, j) in product(1:size(ψ, 1), 1:size(ψ, 2))
        retval[i, j] = @plansor leftenv(ca, i, j, ψ)[1 2; 3] * O[i, j][2 4; 6 5] *
                                ψ.AC[i, j][3 6; 7] * rightenv(ca, i, j, ψ)[7 5; 8] *
                                conj(ψ.AC[i + 1, j][1 4; 8])
    end
    return retval
end

expectation_value(ψ::FiniteQP, O) = expectation_value(convert(FiniteMPS, ψ), O)
function expectation_value(ψ::FiniteQP, O::MPOHamiltonian)
    return expectation_value(convert(FiniteMPS, ψ), O)
end

# more specific typing to account for onsite operators, array of operators, ...
# define expectation_value for MultipliedOperator as scalar multiplication of the non-multiplied result, instead of multiplying the operator itself
function expectation_value(ψ, op::UntimedOperator, args...)
    return op.f * expectation_value(ψ, op.op, args...)
end

# LazySum
# -------
function expectation_value(ψ, ops::LazySum, at::Int)
    return sum(op -> expectation_value(ψ, op, at), ops)
end
function expectation_value(ψ, ops::LazySum, envs::MultipleEnvironments)
    return sum(((op, env),) -> expectation_value(ψ, op, env), zip(ops.ops, envs))
end

# for now we also have LinearCombination
function expectation_value(ψ, H::LinearCombination, envs::LazyLincoCache=environments(ψ, H))
    return return sum(((c, op, env),) -> c * expectation_value(ψ, op, env),
                      zip(H.coeffs, H.opps, envs.envs))
end

# ProjectionOperator
# ------------------
function expectation_value(ψ::FiniteMPS, O::ProjectionOperator,
                           envs::FinEnv=environments(ψ, O))
    ens = zeros(scalartype(ψ), length(ψ))
    for i in 1:length(ψ)
        operator = ∂∂AC(i, ψ, O, envs)
        ens[i] = dot(ψ.AC[i], operator * ψ.AC[i])
    end

    n = norm(ψ.AC[end])^2
    return ens ./ n
end

# Window
# ------
function expectation_value(Ψ::WindowMPS, windowH::Window, windowenvs::WindowEnv)
    left_expval = expectation_value(Ψ.left, windowH.left, windowenvs.left)
    middle_expval = expectation_value(Ψ.middle, windowH.middle, windowenvs.middle)
    right_expval = expectation_value(Ψ.right, windowH.right, windowenvs.right)
    return vcat(left_expval, middle_expval, right_expval)
end

#I need to think about expval with location