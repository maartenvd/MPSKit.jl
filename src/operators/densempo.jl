"""
    FiniteMPO(Os::Vector{<:MPOTensor}) -> FiniteMPO
    FiniteMPO(O::AbstractTensorMap{S,N,N}) where {S,N} -> FiniteMPO

Matrix Product Operator (MPO) acting on a finite tensor product space with a linear order.
"""
struct FiniteMPO{O<:MPOTensor}
    opp::Vector{O}
    function FiniteMPO(Os::Vector{O}) where {O<:MPOTensor}
        for i in eachindex(Os)[1:(end - 1)]
            dual(right_virtualspace(Os[i])) == left_virtualspace(Os[i + 1]) ||
                throw(SpaceMismatch("umatching virtual spaces at site $i"))
        end
        return FiniteMPO{O}(Os)
    end
    function FiniteMPO{O}(Os::Vector{O}) where {O<:MPOTensor}
        return new{O}(Os)
    end
end
function FiniteMPO(O::AbstractTensorMap{S,N,N}) where {S,N}
    return FiniteMPO(decompose_localmpo(add_util_leg(O)))
end

# AbstractVector
# --------------
Base.length(t::FiniteMPO) = length(t.opp)
Base.size(t::FiniteMPO) = (length(t),)

Base.eltype(::FiniteMPO{O}) where {O} = O
Base.eltype(::Type{FiniteMPO{O}}) where {O} = O

Base.getindex(t::FiniteMPO, i) = getindex(t.opp, i)
function Base.setindex!(t::FiniteMPO{O}, v::O, i::Int) where {O}
    @boundscheck begin
        checkbounds(t.opp, i)
        left_virtualspace(v) == left_virtualspace(t, i) &&
            right_virtualspace(v) == right_virtualspace(t, i) ||
            throw(SpaceMismatch("umatching virtual spaces at site $i"))
    end
    @inbounds t.opp[i] = v
    return t
end

# Converters
# ----------
function Base.convert(::Type{<:FiniteMPS}, mpo::FiniteMPO)
    return FiniteMPS(map(mpo.opp) do O
                         @plansor A[-1 -2 -3; -4] := O[-1 -2; 1 2] * τ[1 2; -4 -3]
                     end)
end
function Base.convert(::Type{<:FiniteMPO}, mps::FiniteMPS)
    mpo_tensors = map([mps.AC[1]; mps.AR[2:end]]) do A
        @plansor O[-1 -2; -3 -4] := A[-1 -2 1; 2] * τ[-3 2; -4 1]
    end
    return FiniteMPO(mpo_tensors)
end

# Linear Algebra
# --------------
VectorInterface.scalartype(::Type{FiniteMPO{O}}) where {O} = scalartype(O)

function Base.:+(mpo1::FiniteMPO{TO}, mpo2::FiniteMPO{TO}) where {TO}
    (N = length(mpo1)) == length(mpo2) || throw(ArgumentError("dimension mismatch"))
    @assert left_virtualspace(mpo1, 1) == left_virtualspace(mpo2, 1) &&
            right_virtualspace(mpo1, N) == right_virtualspace(mpo2, N)

    mpo = similar(mpo1.opp)
    halfN = N ÷ 2
    A = storagetype(TO)

    # left half
    F₁ = isometry(A, (right_virtualspace(mpo1, 1) ⊕ right_virtualspace(mpo2, 1))',
                  right_virtualspace(mpo1, 1)')
    F₂ = leftnull(F₁)
    @assert _lastspace(F₂) == right_virtualspace(mpo2, 1)

    @plansor O[-3 -1 -2; -4] := mpo1[1][-1 -2; -3 1] * conj(F₁[-4; 1]) +
                                mpo2[1][-1 -2; -3 1] * conj(F₂[-4; 1])

    # making sure that the new operator is "full rank"
    O, R = leftorth!(O)
    mpo[1] = transpose(O, (2, 3), (1, 4))

    for i in 2:halfN
        # incorporate fusers from left side
        @plansor O₁[-1 -2; -3 -4] := R[-1; 1] * F₁[1; 2] * mpo1[i][2 -2; -3 -4]
        @plansor O₂[-1 -2; -3 -4] := R[-1; 1] * F₂[1; 2] * mpo2[i][2 -2; -3 -4]

        # incorporate fusers from right side
        F₁ = isometry(A, (right_virtualspace(mpo1, i) ⊕ right_virtualspace(mpo2, i))',
                      right_virtualspace(mpo1, i)')
        F₂ = leftnull(F₁)
        @assert _lastspace(F₂) == right_virtualspace(mpo2, i)
        @plansor O[-3 -1 -2; -4] := O₁[-1 -2; -3 1] * conj(F₁[-4; 1]) +
                                    O₂[-1 -2; -3 1] * conj(F₂[-4; 1])

        # making sure that the new operator is "full rank"
        O, R = leftorth!(O)
        mpo[i] = transpose(O, (2, 3), (1, 4))
    end

    C₁, C₂ = F₁, F₂

    # right half
    F₁ = isometry(A, left_virtualspace(mpo1, N) ⊕ left_virtualspace(mpo2, N),
                  left_virtualspace(mpo1, N))
    F₂ = leftnull(F₁)
    @assert _lastspace(F₂) == left_virtualspace(mpo2, N)'

    @plansor O[-1; -3 -4 -2] := F₁[-1; 1] * mpo1[N][1 -2; -3 -4] +
                                F₂[-1; 1] * mpo2[N][1 -2; -3 -4]

    # making sure that the new operator is "full rank"
    L, O = rightorth!(O)
    mpo[end] = transpose(O, (1, 4), (2, 3))

    for i in (N - 1):-1:(halfN + 1)
        # incorporate fusers from right side
        @plansor O₁[-1 -2; -3 -4] := mpo1[i][-1 -2; -3 2] * conj(F₁[1; 2]) * L[1; -4]
        @plansor O₂[-1 -2; -3 -4] := mpo2[i][-1 -2; -3 2] * conj(F₂[1; 2]) * L[1; -4]

        # incorporate fusers from left side
        F₁ = isometry(A, left_virtualspace(mpo1, i) ⊕ left_virtualspace(mpo2, i),
                      left_virtualspace(mpo1, i))
        F₂ = leftnull(F₁)
        @assert _lastspace(F₂) == left_virtualspace(mpo2, i)'
        @plansor O[-1; -3 -4 -2] := F₁[-1; 1] * O₁[1 -2; -3 -4] +
                                    F₂[-1; 1] * O₂[1 -2; -3 -4]

        # making sure that the new operator is "full rank"
        L, O = rightorth!(O)
        mpo[i] = transpose(O, (1, 4), (2, 3))
    end

    # create center gauge and absorb to the right
    C₁ = C₁ * F₁'
    C₂ = C₂ * F₂'
    C = R * (C₁ + C₂) * L
    @plansor mpo[halfN + 1][-1 -2; -3 -4] := mpo[halfN + 1][1 -2; -3 -4] * C[-1; 1]

    return FiniteMPO(mpo)
end

"
    Represents a dense periodic mpo
"
struct DenseMPO{O<:MPOTensor}
    opp::PeriodicArray{O,1}
end

DenseMPO(t::AbstractTensorMap) = DenseMPO(fill(t, 1));
DenseMPO(t::AbstractArray{T,1}) where {T<:MPOTensor} = DenseMPO(PeriodicArray(t));
Base.length(t::DenseMPO) = length(t.opp);
Base.size(t::DenseMPO) = (length(t),)
Base.repeat(t::DenseMPO, n) = DenseMPO(repeat(t.opp, n));
Base.getindex(t::DenseMPO, i) = getindex(t.opp, i);
Base.eltype(::DenseMPO{O}) where {O} = O
VectorInterface.scalartype(::DenseMPO{O}) where {O} = scalartype(O)
Base.iterate(t::DenseMPO, i=1) = (i > length(t.opp)) ? nothing : (t[i], i + 1);
TensorKit.space(t::DenseMPO, i) = space(t.opp[i], 2)
function Base.convert(::Type{InfiniteMPS}, mpo::DenseMPO)
    return InfiniteMPS(map(mpo.opp) do t
                           @plansor tt[-1 -2 -3; -4] := t[-1 -2; 1 2] * τ[1 2; -4 -3]
                       end)
end

function Base.convert(::Type{DenseMPO}, mps::InfiniteMPS)
    return DenseMPO(map(mps.AL) do t
                        @plansor tt[-1 -2; -3 -4] := t[-1 -2 1; 2] * τ[-3 2; -4 1]
                    end)
end

#naively apply the mpo to the mps
function Base.:*(mpo::DenseMPO, st::InfiniteMPS)
    length(st) == length(mpo) || throw(ArgumentError("dimension mismatch"))

    fusers = PeriodicArray(map(zip(st.AL, mpo)) do (al, mp)
                               return isometry(fuse(_firstspace(al), _firstspace(mp)),
                                               _firstspace(al) * _firstspace(mp))
                           end)

    return InfiniteMPS(map(1:length(st)) do i
                           @plansor t[-1 -2; -3] := st.AL[i][1 2; 3] *
                                                    mpo[i][4 -2; 2 5] *
                                                    fusers[i][-1; 1 4] *
                                                    conj(fusers[i + 1][-3; 3 5])
                       end)
end
function Base.:*(mpo::DenseMPO, st::FiniteMPS)
    mod(length(mpo), length(st)) == 0 || throw(ArgumentError("dimension mismatch"))

    tensors = [st.AC[1]; st.AR[2:end]]
    mpot = mpo[1:length(st)]

    fusers = map(zip(tensors, mpot)) do (al, mp)
        return isometry(fuse(_firstspace(al), _firstspace(mp)),
                        _firstspace(al) * _firstspace(mp))
    end

    push!(fusers,
          isometry(fuse(_lastspace(tensors[end])', _lastspace(mpot[end])'),
                   _lastspace(tensors[end])' * _lastspace(mpot[end])'))

    (_firstspace(mpot[1]) == oneunit(_firstspace(mpot[1])) &&
     _lastspace(mpot[end])' == _firstspace(mpot[1])) ||
        @warn "mpo does not start/end with a trivial leg"

    return FiniteMPS(map(1:length(st)) do i
                         @plansor t[-1 -2; -3] := tensors[i][1 2; 3] *
                                                  mpot[i][4 -2; 2 5] *
                                                  fusers[i][-1; 1 4] *
                                                  conj(fusers[i + 1][-3; 3 5])
                     end)
end

function Base.:*(mpo1::DenseMPO, mpo2::DenseMPO)
    length(mpo1) == length(mpo2) || throw(ArgumentError("dimension mismatch"))

    fusers = PeriodicArray(map(zip(mpo2.opp, mpo1.opp)) do (mp1, mp2)
                               return isometry(fuse(_firstspace(mp1), _firstspace(mp2)),
                                               _firstspace(mp1) * _firstspace(mp2))
                           end)

    return DenseMPO(map(1:length(mpo1)) do i
                        @plansor t[-1 -2; -3 -4] := mpo2[i][1 2; -3 3] *
                                                    mpo1[i][4 -2; 2 5] *
                                                    fusers[i][-1; 1 4] *
                                                    conj(fusers[i + 1][-4; 3 5])
                    end)
end

function TensorKit.dot(a::InfiniteMPS, mpo::DenseMPO, b::InfiniteMPS; krylovdim=30)
    init = similar(a.AL[1],
                   _firstspace(b.AL[1]) * _firstspace(mpo.opp[1]) ← _firstspace(a.AL[1]))
    randomize!(init)

    val, = fixedpoint(TransferMatrix(b.AL, mpo.opp, a.AL), init, :LM,
                      Arnoldi(; krylovdim=krylovdim))
    return val
end
