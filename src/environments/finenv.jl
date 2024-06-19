"""
    FinEnv keeps track of the environments for FiniteMPS
    It automatically checks if the queried environment is still correctly cached and if not - recalculates

    if above is set to nothing, above === below.

    opp can be a vector of nothing, in which case it'll just be the overlap
"""
struct FinEnv{A,B,C,D} <: Cache
    above::A

    opp::B #the operator

    ldependencies::Vector{C} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Vector{C}

    leftenvs::Vector{D}
    rightenvs::Vector{D}
end

#do we want to call these function environments? or do we want to restrict it to
# only the constructor environments(state,ham)?

function environments(below, t::Tuple, args...; kwargs...)
    return environments(below, t[1], t[2], args...; kwargs...)
end;
function environments(below, opp, leftstart, rightstart)
    return environments(below, opp, nothing, leftstart, rightstart)
end;
function environments(below, opp, above, leftstart, rightstart)
    leftenvs = [leftstart]
    rightenvs = [rightstart]

    for i in 1:length(below)
        push!(leftenvs, similar(leftstart))
        push!(rightenvs, similar(rightstart))
    end
    t = similar(below.AL[1])
    return FinEnv(above, opp, fill(t, length(below)), fill(t, length(below)), leftenvs,
                  reverse(rightenvs))
end

#automatically construct the correct leftstart/rightstart for a finitemps
function environments(below::FiniteMPS{S}, O::Union{SparseMPO,MPOHamiltonian},
                      above=nothing) where {S}
    lll = l_LL(below)
    rrr = r_RR(below)
    rightstart = Vector{S}()
    leftstart = Vector{S}()

    for i in 1:(O.odim)
        util_left = Tensor(x -> storagetype(S)(undef, x), O.domspaces[1, i]')
        fill_data!(util_left, one)
        util_right = Tensor(x -> storagetype(S)(undef, x), O.imspaces[length(below), i]')
        fill_data!(util_right, one)

        @plansor ctl[-1 -2; -3] := lll[-1; -3] * util_left[-2]
        @plansor ctr[-1 -2; -3] := rrr[-1; -3] * util_right[-2]

        if i != 1
            ctl = zero(ctl)
        end

        if (i != O.odim && O isa MPOHamiltonian) || (i != 1 && O isa SparseMPO)
            ctr = zero(ctr)
        end

        push!(leftstart, ctl)
        push!(rightstart, ctr)
    end

    return environments(below, O, above, leftstart, rightstart)
end

function MPSKit.environments(below::FiniteMPS{S}, O::DenseMPO, above=nothing) where {S}
    N = length(below)
    leftstart = isomorphism(storagetype(S),
                            left_virtualspace(below, 0) ⊗ space(O[1], 1)' ←
                            left_virtualspace(below, 0))
    rightstart = isomorphism(storagetype(S),
                             right_virtualspace(below, N) ⊗ space(O[N], 4)' ←
                             right_virtualspace(below, length(below)))
    return environments(below, O, above, leftstart, rightstart)
end

function environments(below::FiniteMPS, above::FiniteMPS)
    opp = fill(nothing, length(below))
    return environments(below, opp, above, l_LL(above), r_RR(above))
end

function environments(state::Union{FiniteMPS,WindowMPS}, opp::ProjectionOperator)
    @plansor leftstart[-1; -2 -3 -4] := l_LL(opp.ket)[-3; -4] * l_LL(opp.ket)[-1; -2]
    @plansor rightstart[-1; -2 -3 -4] := r_RR(opp.ket)[-1; -2] * r_RR(opp.ket)[-3; -4]
    return environments(state, fill(nothing, length(state)), state, leftstart, rightstart)
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
# this forces the transfers to be recalculated lazily 
function invalidate!(ca::FinEnv, ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    return ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

#rightenv[ind] will be contracteable with the tensor on site [ind]
function rightenv(ca::FinEnv, ind, state)
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]), length(state):-1:(ind + 1))
    a = isnothing(a) ? nothing : length(state) - a + 1

    if !isnothing(a)
        #we need to recalculate
        for j in a:-1:(ind + 1)
            above = isnothing(ca.above) ? state.AR[j] : ca.above.AR[j]
            ca.rightenvs[j] = TransferMatrix(above, ca.opp[j], state.AR[j]) *
                              ca.rightenvs[j + 1]
            ca.rdependencies[j] = state.AR[j]
        end
    end

    return ca.rightenvs[ind + 1]
end

function leftenv(ca::FinEnv, ind, state)
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]), 1:(ind - 1))

    if !isnothing(a)
        #we need to recalculate
        for j in a:(ind - 1)
            above = isnothing(ca.above) ? state.AL[j] : ca.above.AL[j]
            ca.leftenvs[j + 1] = ca.leftenvs[j] *
                                 TransferMatrix(above, ca.opp[j], state.AL[j])
            ca.ldependencies[j] = state.AL[j]
        end
    end

    return ca.leftenvs[ind]
end
