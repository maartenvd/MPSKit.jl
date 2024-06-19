"calculates the entropy of a given state"
entropy(state::InfiniteMPS) = map(c -> -tr(safe_xlogx(c * c')), state.CR);
function entropy(state::Union{FiniteMPS,WindowMPS,InfiniteMPS}, loc::Int)
    return -tr(safe_xlogx(state.CR[loc] * state.CR[loc]'))
end;

function infinite_temperature(H::MPOHamiltonian)
    return [permute(isomorphism(storagetype(H[1, 1, 1]), oneunit(sp) * sp,
                                oneunit(sp) * sp), (1, 2, 4), (3,)) for sp in H.pspaces]
end

"""
    calc_galerkin(state, envs)

Calculate the galerkin error.
"""
function calc_galerkin(state::Union{InfiniteMPS,FiniteMPS,WindowMPS}, loc, envs)::Float64
    AC´ = ∂∂AC(loc, state, envs.opp, envs) * state.AC[loc]
    normalize!(AC´)
    out = add!(AC´, state.AL[loc] * state.AL[loc]' * AC´, -1)
    return norm(out)
end
function calc_galerkin(state::Union{InfiniteMPS,FiniteMPS,WindowMPS}, envs)::Float64
    return maximum([calc_galerkin(state, loc, envs) for loc in 1:length(state)])
end
function calc_galerkin(state::MPSMultiline, envs::PerMPOInfEnv)::Float64
    above = isnothing(envs.above) ? state : envs.above

    εs = zeros(Float64, size(state, 1), size(state, 2))
    for (row, col) in product(1:size(state, 1), 1:size(state, 2))
        AC´ = ∂∂AC(row, col, state, envs.opp, envs) * above.AC[row, col]
        normalize!(AC´)
        out = add!(AC´, state.AL[row + 1, col] * state.AL[row + 1, col]' * AC´, -1)
        εs[row, col] = norm(out)
    end

    return maximum(εs[:])
end

"
Calculates the (partial) transfer spectrum
"
function transfer_spectrum(above::InfiniteMPS; below=above, tol=Defaults.tol, num_vals=20,
                           sector=first(sectors(oneunit(left_virtualspace(above, 1)))))
    init = randomize!(similar(above.AL[1], left_virtualspace(below, 0),
                              ℂ[typeof(sector)](sector => 1)' * left_virtualspace(above, 0)))

    transferspace = fuse(left_virtualspace(above, 0) * left_virtualspace(below, 0)')
    num_vals = min(dim(transferspace, sector), num_vals) # we can ask at most this many values
    eigenvals, eigenvecs, convhist = eigsolve(flip(TransferMatrix(above.AL, below.AL)),
                                              init, num_vals, :LM; tol=tol)
    convhist.converged < num_vals &&
        @warn "correlation length failed to converge: normres = $(convhist.normres)"

    return eigenvals
end

"""
    entanglement_spectrum(ψ; site::Int=0) -> SectorDict{sectortype(ψ),Vector{<:Real}}

Compute the entanglement spectrum at a given site, i.e. the singular values of the gauge
matrix to the right of a given site. This is a dictionary mapping the charge to the singular
value.
"""
function entanglement_spectrum(st::Union{InfiniteMPS,FiniteMPS,WindowMPS}, site::Int=0)
    @assert site <= length(st)
    _, S, = tsvd(st.CR[site])
    return TensorKit.SectorDict(c => real(diag(b)) for (c, b) in blocks(S))
end

"""
Find the closest fractions of π, differing at most ```tol_angle```
"""
function approx_angles(spectrum; tol_angle=0.1)
    angles = angle.(spectrum) ./ π                          # ∈ ]-1, 1]
    angles_approx = rationalize.(angles, tol=tol_angle)     # ∈ [-1, 1]

    # Remove the effects of the branchcut.
    angles_approx[findall(angles_approx .== -1)] .= 1       # ∈ ]-1, 1]

    return angles_approx .* π                               # ∈ ]-π, π]
end

"""
Given an InfiniteMPS, compute the gap ```ϵ``` for the asymptotics of the transfer matrix, as well as the Marek gap ```δ``` as a scaling measure of the bond dimension.
"""
function marek_gap(above::InfiniteMPS; tol_angle=0.1, kwargs...)
    spectrum = transfer_spectrum(above; kwargs...)
    return marek_gap(spectrum; tol_angle)
end

function marek_gap(spectrum; tol_angle=0.1)
    # Remove 1s from the spectrum
    inds = findall(abs.(spectrum) .< 1 - 1e-12)
    length(spectrum) - length(inds) < 2 || @warn "Non-injective mps?"

    spectrum = spectrum[inds]

    angles = approx_angles(spectrum; tol_angle=tol_angle)
    θ = first(angles)

    spectrum_at_angle = spectrum[findall(angles .== θ)]

    lambdas = -log.(abs.(spectrum_at_angle))

    ϵ = first(lambdas)

    δ = Inf
    if length(lambdas) > 2
        δ = lambdas[2] - lambdas[1]
    end

    return ϵ, δ, θ
end

"""
Compute the correlation length of a given InfiniteMPS.
"""
function correlation_length(above::InfiniteMPS; kwargs...)
    ϵ, = marek_gap(above; kwargs...)
    return 1 / ϵ
end

function correlation_length(spectrum; kwargs...)
    ϵ, = marek_gap(spectrum; kwargs...)
    return 1 / ϵ
end

"""
    variance(state, hamiltonian, [envs=environments(state, hamiltonian)])

Compute the variance of the energy of the state with respect to the hamiltonian.
"""
function variance end

function variance(state::InfiniteMPS, H::MPOHamiltonian, envs=environments(state, H))
    rescaled_H = H - expectation_value(state, H, envs)
    return real(sum(expectation_value(state, rescaled_H * rescaled_H)))
end

function variance(state::FiniteMPS, H::MPOHamiltonian, envs=environments(state, H))
    H2 = H * H
    return real(sum(expectation_value(state, H2)) -
                sum(expectation_value(state, H, envs))^2)
end

function variance(state::FiniteQP, H::MPOHamiltonian, args...)
    return variance(convert(FiniteMPS, state), H)
end;

function variance(state::InfiniteQP, H::MPOHamiltonian, envs=environments(state, H))
    # I remember there being an issue here @gertian?
    state.trivial ||
        throw(ArgumentError("variance of domain wall excitations is not implemented"))

    rescaled_H = H - expectation_value(state.left_gs, H)

    #I don't remember where the formula came from
    E_ex = dot(state, effective_excitation_hamiltonian(H, state, envs))
    E_f = expectation_value(state.left_gs, rescaled_H, 1:0)

    H2 = rescaled_H * rescaled_H

    return real(dot(state, effective_excitation_hamiltonian(H2, state)) -
                2 * (E_f + E_ex) * E_ex + E_ex^2)
end

function variance(ψ, H::LazySum, envs=environments(ψ, sum(H)))
    envs isa MultipleEnvironments &&
        throw(ArgumentError("The environment cannot be Lazy i.e. use environments of sum(H)"))
    return variance(ψ, sum(H), envs)
end

function variance(ψ::WindowMPS, H::Union{MPOHamiltonian,Window}, envs=environments(ψ, H))
    #tricky to define
    H2, nenvs = squaredenvs(ψ, H, envs)
    return real(expectation_value(ψ, H2, 1:length(ψ), nenvs) -
                expectation_value(ψ, H, 1:length(ψ), envs)^2)
end

"""
You can impose periodic boundary conditions on an mpo-hamiltonian (for a given size)
That creates a new mpo-hamiltonian with larger bond dimension
The interaction never wraps around multiple times
"""
function periodic_boundary_conditions(H::MPOHamiltonian{S,T,E},
                                      len=H.period) where {S,T,E}
    sanitycheck(H) || throw(ArgumentError("invalid hamiltonian"))
    mod(len, H.period) == 0 ||
        throw(ArgumentError("$(len) is not a multiple of unitcell"))

    fusers = PeriodicArray(map(1:len) do loc
                               map(Iterators.product(H.domspaces[loc, :],
                                                     H.domspaces[len + 1, :],
                                                     H.domspaces[loc, :])) do (v1, v2,
                                                                               v3)
                                   return isomorphism(storagetype(T), fuse(v1 * v2' * v3),
                                                      v1 * v2' * v3)
                               end
                           end)

    #a -> what progress have I made in the upper layer?
    #b -> what virtual space did I "lend" in the beginning?
    #c -> what progress have I made in the lower layer?
    χ = H.odim

    indmap = zeros(Int, χ, χ, χ)
    χ´ = 0
    for b in χ:-1:2, c in b:χ
        χ´ += 1
        indmap[1, b, c] = χ´
    end

    for a in 2:χ, b in χ:-1:a
        χ´ += 1
        indmap[a, b, χ] = χ´
    end

    #do the bulk
    bulk = PeriodicArray(convert(Array{Union{T,E},3}, fill(zero(E), H.period, χ´, χ´)))

    for loc in 1:(H.period), (j, k) in keys(H[loc])

        #apply (j,k) above
        l = H.odim
        for i in 2:(H.odim)
            k <= i && i <= l || continue

            f1 = fusers[loc][j, i, l]
            f2 = fusers[loc + 1][k, i, l]
            j′ = indmap[j, i, l]
            k′ = indmap[k, i, l]
            @plansor bulk[loc, j′, k′][-1 -2; -3 -4] := H[loc][j, k][1 2; -3 6] *
                                                        f1[-1; 1 3 5] *
                                                        conj(f2[-4; 6 7 8]) * τ[2 3; 7 4] *
                                                        τ[4 5; 8 -2]
        end

        #apply (j,k) below
        i = 1
        for l in 2:(H.odim - 1)
            l > 1 && l >= i && l <= j || continue

            f1 = fusers[loc][i, l, j]
            f2 = fusers[loc + 1][i, l, k]
            j′ = indmap[i, l, j]
            k′ = indmap[i, l, k]
            @plansor bulk[loc, j′, k′][-1 -2; -3 -4] := H[loc][j, k][1 -2; 3 6] *
                                                        f1[-1; 4 2 1] *
                                                        conj(f2[-4; 8 7 6]) * τ[5 2; 7 3] *
                                                        τ[-3 4; 8 5]
        end
    end

    # make the starter
    starter = convert(Array{Union{T,E},2}, fill(zero(E), χ´, χ´))
    for (j, k) in keys(H[1])

        #apply (j,k) above
        if j == 1
            f1 = fusers[1][1, end, end]
            f2 = fusers[2][k, end, end]
            j′ = indmap[k, H.odim, H.odim]
            @plansor starter[1, j′][-1 -2; -3 -4] := H[1][j, k][-1 -2; -3 2] *
                                                     conj(f2[-4; 2 3 3])
        end

        #apply (j,k) below
        if j > 1 && j < H.odim
            f1 = fusers[1][1, j, j]
            f2 = fusers[2][1, j, k]

            @plansor starter[1, indmap[1, j, k]][-1 -2; -3 -4] := H[1][j, k][4 -2; 3 1] *
                                                                  conj(f2[-4; 6 2 1]) *
                                                                  τ[5 4; 2 3] *
                                                                  τ[-3 -1; 6 5]
        end
    end
    starter[1, 1] = one(E)
    starter[end, end] = one(E)

    # make the ender
    ender = convert(Array{Union{T,E},2}, fill(zero(E), χ´, χ´))
    for (j, k) in keys(H[H.period])
        if k > 1
            f1 = fusers[end][j, k, H.odim]
            k′ = indmap[j, k, H.odim]
            @plansor ender[k′, end][-1 -2; -3 -4] := f1[-1; 1 2 6] *
                                                     H[H.period][j, k][1 3; -3 4] *
                                                     τ[3 2; 4 5] * τ[5 6; -4 -2]
        end
    end
    ender[1, 1] = one(E)
    ender[end, end] = one(E)

    # fill in the entire H
    nos = convert(Array{Union{T,E},3}, fill(zero(E), len, χ´, χ´))
    nos[1, :, :] = starter[:, :]
    nos[end, :, :] = ender[:, :]

    for i in 2:(len - 1)
        nos[i, :, :] = bulk[i, :, :]
    end

    return MPOHamiltonian(nos)
end

#impose periodic boundary conditions on a normal mpo
function periodic_boundary_conditions(mpo::DenseMPO{O}, len=length(mpo)) where {O}
    mod(len, length(mpo)) == 0 || throw(ArgumentError("len not a multiple of unitcell"))

    output = PeriodicArray{O,1}(undef, len)

    sp = _firstspace(mpo[1])'
    utleg = fill_data!(similar(mpo[1], oneunit(sp)), one)

    #do the bulk
    for j in 2:(len - 1)
        f1 = isomorphism(storagetype(O), fuse(sp * _firstspace(mpo[j])),
                         sp * _firstspace(mpo[j]))
        f2 = isomorphism(storagetype(O), fuse(sp * _lastspace(mpo[j])'),
                         sp * _lastspace(mpo[j])')

        @plansor output[j][-1 -2; -3 -4] := mpo[j][2 -2; 3 5] * f1[-1; 1 2] *
                                            conj(f2[-4; 4 5]) * τ[-3 1; 4 3]
    end

    #do the left
    f2 = isomorphism(storagetype(O), fuse(sp * _lastspace(mpo[1])'),
                     sp * _lastspace(mpo[1])')
    @plansor output[1][-1 -2; -3 -4] := mpo[1][1 -2; 3 5] * conj(f2[-4; 4 5]) *
                                        τ[-3 1; 4 3] * utleg[-1]

    #do the right
    f2 = isomorphism(storagetype(O), fuse(sp * _firstspace(mpo[len])),
                     sp * _firstspace(mpo[len]))
    @plansor output[end][-1 -2; -3 -4] := mpo[len][2 -2; 3 4] * f2[-1; 1 2] * τ[-3 1; 4 3] *
                                          conj(utleg[-4])

    return DenseMPO(output)
end
