#===========================================================================================
Tensor types
===========================================================================================#

const MPOTensor{S} = AbstractTensorMap{S,2,2} where {S}
const MPSBondTensor{S} = AbstractTensorMap{S,1,1} where {S}
const GenericMPSTensor{S,N} = AbstractTensorMap{S,N,1} where {S,N} #some functions are also defined for "general mps tensors" (used in peps code)
const MPSTensor{S} = GenericMPSTensor{S,2} where {S} #the usual mps tensors on which we work
#const ExMPSTensor{S,N,A,G,F1,F2}=GenericMPSTensor{S,3,A,G,F1,F2} #and mps tensor with an extra excitation - utility leg

"""
    MPSTensor([f, eltype], d::Int, left_D::Int, [right_D]::Int])
    MPSTensor([f, eltype], physicalspace::Union{S,CompositeSpace{S}}, 
              left_virtualspace::S, [right_virtualspace]::S) where {S<:ElementarySpace}

Construct an `MPSTensor` with given physical and virtual spaces.

### Arguments
- `f::Function=rand`: initializer function for tensor data
- `eltype::Type{<:Number}=ComplexF64`: scalar type of tensors

- `physicalspace::Union{S,CompositeSpace{S}}`: physical space
- `left_virtualspace::S`: left virtual space
- `right_virtualspace::S`: right virtual space, defaults to equal left

- `d::Int`: physical dimension
- `left_D::Int`: left virtual dimension
- `right_D::Int`: right virtual dimension
"""
function MPSTensor(f, eltype, P::Union{S,CompositeSpace{S}}, Vₗ::S,
                   Vᵣ::S=Vₗ) where {S<:ElementarySpace}
    return TensorMap(f, eltype, Vₗ ⊗ P ← Vᵣ)
end
function MPSTensor(P::Union{S,CompositeSpace{S}}, Vₗ::S,
                   Vᵣ::S=Vₗ) where {S<:ElementarySpace}
    return MPSTensor(rand, Defaults.eltype, P, Vₗ, Vᵣ)
end

"""
    MPSTensor([f, eltype], d::Int, Dₗ::Int, [Dᵣ]::Int])

Construct an `MPSTensor` with given physical and virtual dimensions.

### Arguments
- `f::Function=rand`: initializer function for tensor data
- `eltype::Type{<:Number}=ComplexF64`: scalar type of tensors
- `d::Int`: physical dimension
- `Dₗ::Int`: left virtual dimension
- `Dᵣ::Int`: right virtual dimension
"""
MPSTensor(f, eltype, d::Int, Dₗ::Int, Dᵣ::Int=Dₗ) = MPSTensor(f, eltype, ℂ^d, ℂ^Dₗ, ℂ^Dᵣ)
MPSTensor(d::Int, Dₗ::Int, Dᵣ::Int=Dₗ) = MPSTensor(ℂ^d, ℂ^Dₗ, ℂ^Dᵣ)

"""
    MPSTensor(A::AbstractArray)

Convert an array to an `MPSTensor`.
"""
function MPSTensor(A::AbstractArray{T}) where {T<:Number}
    @assert ndims(A) > 2 "MPSTensor should have at least 3 dims, but has $ndims(A)"
    sz = size(A)
    t = TensorMap(undef, T, foldl(⊗, ComplexSpace.(sz[1:(end - 1)])) ← ℂ^sz[end])
    t[] .= A
    return t
end

"""
    isfullrank(A::GenericMPSTensor)

Determine whether the given tensor is full rank, i.e. whether both the map from the left
virtual space and the physical space to the right virtual space, and the map from the right
virtual space and the physical space to the left virtual space are injective.
"""
function isfullrank(A::GenericMPSTensor)
    Vₗ = _firstspace(A)
    Vᵣ = _lastspace(A)
    P = ⊗(space.(Ref(A), 2:(numind(A) - 1))...)
    return Vₗ ⊗ P ≿ Vᵣ' && Vₗ' ≾ P ⊗ Vᵣ
end

left_virtualspace(A::GenericMPSTensor) = _firstspace(A)
right_virtualspace(A::GenericMPSTensor) = _lastspace(A)

#===========================================================================================
MPS types
===========================================================================================#

abstract type AbstractMPS end

Base.eltype(ψ::AbstractMPS) = eltype(typeof(ψ))
VectorInterface.scalartype(T::Type{<:AbstractMPS}) = scalartype(site_type(T))

"""
    site_type(ψ::AbstractMPS)
    site_type(ψtype::Type{<:AbstractMPS})

Return the type of the site tensors of an `AbstractMPS`.
"""
site_type(ψ::AbstractMPS) = site_type(typeof(ψ))

"""
    bond_type(ψ::AbstractMPS)
    bond_type(ψtype::Type{<:AbstractMPS})

Return the type of the bond tensors of an `AbstractMPS`.
"""
bond_type(ψ::AbstractMPS) = bond_type(typeof(ψ))

TensorKit.spacetype(ψ::AbstractMPS) = spacetype(typeof(ψ))
TensorKit.spacetype(ψtype::Type{<:AbstractMPS}) = spacetype(site_type(ψtype))
TensorKit.sectortype(ψ::AbstractMPS) = sectortype(typeof(ψ))
TensorKit.sectortype(ψtype::Type{<:AbstractMPS}) = sectortype(site_type(ψtype))

"""
    left_virtualspace(ψ::AbstractMPS, i::Int)
    
Return the left virtual space of the bond tensor at site `i`. This is equivalent to the
left virtual space of the left-gauged site tensor at site `i + 1`.
"""
function left_virtualspace end

"""
    right_virtualspace(ψ::AbstractMPS, i::Int)

Return the right virtual space of the bond tensor at site `i`. This is equivalent to the
right virtual space of the right-gauged site tensor at site `i`.
"""
function right_virtualspace end

"""
    physicalspace(ψ::AbstractMPS, i::Int)

Return the physical space of the site tensor at site `i`.
"""
function physicalspace end

abstract type AbstractFiniteMPS <: AbstractMPS end
