const WINDOW_FIXED = :F
const WINDOW_VARIABLE = :V

"""
    WindowMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractFiniteMPS

Type that represents a finite Matrix Product State embedded in an infinite Matrix Product State.

## Fields

- `left::InfiniteMPS` -- left infinite state
- `middle::FiniteMPS` -- finite state in the middle
- `right::InfiniteMPS` -- right infinite state

---

## Constructors

    WindowMPS(left::InfiniteMPS, middle::FiniteMPS, right::InfiniteMPS)
    WindowMPS(left::InfiniteMPS, middle_tensors::AbstractVector, right::InfiniteMPS)
    WindowMPS([f, eltype], physicalspaces::Vector{<:Union{S, CompositeSpace{S}},
              virtualspaces::Vector{<:Union{S, CompositeSpace{S}}, left::InfiniteMPS,
              right_gs::InfiniteMPS)
    WindowMPS([f, eltype], physicalspaces::Vector{<:Union{S,CompositeSpace{S}}},
              maxvirtualspace::S, left::InfiniteMPS, right_gs::InfiniteMPS)
    
Construct a WindowMPS via a specification of left and right infinite state, and either
a middle state or a vector of tensors to construct the middle. Alternatively, it is possible
to supply the same arguments as for the constructor of [`FiniteMPS`](@ref), followed by a
left and right state to construct the WindowMPS in one step.

    WindowMPS(state::InfiniteMPS, L::Int; copyright=false)

Construct a WindowMPS from an InfiniteMPS, by promoting a region of length `L` to a
`FiniteMPS`. Note that by default the right state is not copied (and thus .left === .right).

Options for fixing the left and right infinite state (i.e. so they don't get time evolved) 
can be done via the Boolean keyword arguments `fixleft` and `fixright`.
"""
struct WindowMPS{A<:GenericMPSTensor,B<:MPSBondTensor,VL,VR} <: AbstractFiniteMPS
    window::Window{InfiniteMPS{A,B},FiniteMPS{A,B},InfiniteMPS{A,B}}

    function WindowMPS(ψₗ::InfiniteMPS{A,B}, ψₘ::FiniteMPS{A,B},
                       ψᵣ::InfiniteMPS{A,B}; fixleft::Bool=false,
                       fixright::Bool=false) where {A,B}
        left_virtualspace(ψₗ, 0) == left_virtualspace(ψₘ, 0) &&
            right_virtualspace(ψₘ, length(ψₘ)) == right_virtualspace(ψᵣ, length(ψₘ)) ||
            throw(SpaceMismatch("Mismatch between window and environment virtual spaces"))
        VL = fixleft ? WINDOW_FIXED : WINDOW_VARIABLE
        VR = fixright ? WINDOW_FIXED : WINDOW_VARIABLE
        return new{A,B,VL,VR}(Window(ψₗ, ψₘ, ψᵣ))
    end
end

#===========================================================================================
Constructors
===========================================================================================#
function WindowMPS(ψₗ::InfiniteMPS, site_tensors::AbstractVector{<:GenericMPSTensor},
                   ψᵣ::InfiniteMPS; kwargs...)
    return WindowMPS(ψₗ, FiniteMPS(site_tensors), ψᵣ; kwargs...)
end

#perhaps we want to consider not using the FiniteMPS constructor since I believe these constructs
#the spaces so that the edge virtual sapces are one dimensional.
function WindowMPS(f, elt, physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                   maxvirtspace::S, ψₗ::InfiniteMPS,
                   ψᵣ::InfiniteMPS; kwargs...) where {S<:ElementarySpace}
    ψₘ = FiniteMPS(f, elt, physspaces, maxvirtspace; left=left_virtualspace(ψₗ, 0),
                   right=right_virtualspace(ψᵣ, length(physspaces)))
    return WindowMPS(ψₗ, ψₘ, ψᵣ; kwargs...)
end
function WindowMPS(physspaces::Vector{<:Union{S,CompositeSpace{S}}}, maxvirtspace::S,
                   ψₗ::InfiniteMPS, ψᵣ::InfiniteMPS; kwargs...) where {S<:ElementarySpace}
    return WindowMPS(rand, Defaults.eltype, physspaces, maxvirtspace, ψₗ, ψᵣ; kwargs...)
end

function WindowMPS(f, elt, physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                   virtspaces::Vector{S}, ψₗ::InfiniteMPS,
                   ψᵣ::InfiniteMPS; kwargs...) where {S<:ElementarySpace}
    ψₘ = FiniteMPS(f, elt, physspaces, virtspaces)
    return WindowMPS(ψₗ, ψₘ, ψᵣ; kwargs...)
end
function WindowMPS(physspaces::Vector{<:Union{S,CompositeSpace{S}}}, virtspaces::Vector{S},
                   ψₗ::InfiniteMPS, ψᵣ::InfiniteMPS; kwargs...) where {S<:ElementarySpace}
    return WindowMPS(rand, Defaults.eltype, physspaces, virtspaces, ψₗ, ψᵣ; kwargs...)
end

function WindowMPS(f, elt, P::ProductSpace, args...; kwargs...)
    return WindowMPS(f, elt, collect(P), args...; kwargs...)
end
function WindowMPS(P::ProductSpace, args...; kwargs...)
    return WindowMPS(collect(P), args...; kwargs...)
end

function WindowMPS(f, elt, N::Int, V::VectorSpace, args...; kwargs...)
    return WindowMPS(f, elt, fill(V, N), args...; kwargs...)
end
function WindowMPS(N::Int, V::VectorSpace, args...; kwargs...)
    return WindowMPS(fill(V, N), args...; kwargs...)
end

function WindowMPS(ψ::InfiniteMPS{A,B}, L::Int; copyright=false, kwargs...) where {A,B}
    CLs = Vector{Union{Missing,B}}(missing, L + 1)
    ALs = Vector{Union{Missing,A}}(missing, L)
    ARs = Vector{Union{Missing,A}}(missing, L)
    ACs = Vector{Union{Missing,A}}(missing, L)

    ALs .= ψ.AL[1:L]
    ARs .= ψ.AR[1:L]
    ACs .= ψ.AC[1:L]
    CLs .= ψ.CR[0:L]

    return WindowMPS(ψ, FiniteMPS(ALs, ARs, ACs, CLs), copyright ? copy(ψ) : ψ; kwargs...)
end

#===========================================================================================
Utility
===========================================================================================#
function Base.getproperty(ψ::WindowMPS, sym::Symbol)
    if sym === :left || sym === :middle || sym === :right
        return getfield(ψ.window, sym)
    elseif sym === :AL
        return ALView(ψ)
    elseif sym === :AR
        return ARView(ψ)
    elseif sym === :AC
        return ACView(ψ)
    elseif sym === :CR
        return CRView(ψ)
    else
        return getfield(ψ, sym)
    end
end

function Base.copy(ψ::WindowMPS{A,B,VL,VR}) where {A,B,VL,VR}
    left = VL === WINDOW_VARIABLE ? copy(ψ.left) : ψ.left
    fixleft = VL === WINDOW_VARIABLE ? false : true
    right = VR === WINDOW_VARIABLE ? copy(ψ.right) : ψ.right
    fixright = VR === WINDOW_VARIABLE ? false : true
    return WindowMPS(left, copy(ψ.middle), right; fixleft=fixleft, fixright=fixright)
end

# not sure about the underlying methods...
Base.length(ψ::WindowMPS) = length(ψ.middle)
Base.size(ψ::WindowMPS, i...) = size(ψ.middle, i...)
Base.eltype(::Type{<:WindowMPS{A}}) where {A} = A

site_type(::Type{<:WindowMPS{A}}) where {A} = A
bond_type(::Type{<:WindowMPS{<:Any,B}}) where {B} = B

TensorKit.space(ψ::WindowMPS, n::Integer) = space(ψ.AC[n], 2)
left_virtualspace(ψ::WindowMPS, n::Integer) = left_virtualspace(ψ.middle, n);
right_virtualspace(ψ::WindowMPS, n::Integer) = right_virtualspace(ψ.middle, n);

r_RR(ψ::WindowMPS) = r_RR(ψ.right, length(ψ))
l_LL(ψ::WindowMPS) = l_LL(ψ.left, 1)

max_Ds(ψ::WindowMPS) = max_Ds(ψ.middle)

Base.:*(ψ::WindowMPS, a::Number) = rmul!(copy(ψ), a)
Base.:*(a::Number, ψ::WindowMPS) = lmul!(a, copy(ψ))

function TensorKit.lmul!(a::Number, ψ::WindowMPS)
    lmul!(a, ψ.middle)
    return ψ
end

function TensorKit.rmul!(ψ::WindowMPS, a::Number)
    rmul!(ψ.middle, a)
    return ψ
end

function TensorKit.dot(ψ₁::WindowMPS, ψ₂::WindowMPS)
    length(ψ₁) == length(ψ₂) || throw(ArgumentError("MPS with different length"))
    ψ₁.left == ψ₂.left ||
        dot(ψ₁.left, ψ₂.left) ≈ 1 ||
        throw(ArgumentError("left InfiniteMPS are different"))
    ψ₁.right == ψ₂.right ||
        dot(ψ₁.right, ψ₂.right) ≈ 1 ||
        throw(ArgumentError("right InfiniteMPS are different"))

    ρr = TransferMatrix(ψ₂.AR[2:end], ψ₁.AR[2:end]) * r_RR(ψ₂)
    return tr(_transpose_front(ψ₁.AC[1])' * _transpose_front(ψ₂.AC[1]) * ρr)
end

TensorKit.norm(ψ::WindowMPS) = norm(ψ.middle)

TensorKit.normalize!(ψ::WindowMPS) = normalize!(ψ.middle)
TensorKit.normalize(ψ::WindowMPS) = normalize!(copy(ψ))
