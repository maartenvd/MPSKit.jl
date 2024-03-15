"""
    InfiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbtractMPS

Type that represents an infinite Matrix Product State.

## Fields
- `AL` -- left-gauged MPS tensors
- `AR` -- right-gauged MPS tensors
- `AC` -- center-gauged MPS tensors
- `CR` -- gauge tensors

## Notes
By convention, we have that:
- `AL[i] * CR[i]` = `AC[i]` = `CR[i-1] * AR[i]`
- `AL[i]' * AL[i] = 1`
- `AR[i] * AR[i]' = 1`

---

## Constructors
    InfiniteMPS([f, eltype], physicalspaces::Vector{<:Union{S, CompositeSpace{S}},
                virtualspaces::Vector{<:Union{S, CompositeSpace{S}};
                kwargs...) where {S<:ElementarySpace}
    InfiniteMPS(As::AbstractVector{<:GenericMPSTensor}; kwargs...)
    InfiniteMPS(AL, AR, CR, [AC]) # expert

Construct and gauge an MPS via a specification of physical and virtual spaces, or from a list of
tensors `As`. Alternatively, just supply the fields `AL`, `AR`, `CR`, and `AC` directly, but
beware that no gauging will be performed in that case.

### Arguments
- `As::AbstractVector{<:GenericMPSTensor}`: vector of site tensors
- `ALs::AbstractVector{<:GenericMPSTensor}`: vector of left-gauged site tensors
- `C₀::MPSBondTensor`: initial gauge tensor

- `f::Function=rand`: initializer function for tensor data
- `eltype::Type{<:Number}=ComplexF64`: scalar type of tensors

- `physicalspaces::AbstractVector{<:Union{S, CompositeSpace{S}}`: list of physical spaces
- `virtualspaces::AbstractVector{<:Union{S, CompositeSpace{S}}`: list of virtual spaces

### Keywords
- `tol`: gauge fixing tolerance
- `maxiter`: gauge fixing maximum iterations
"""
struct InfiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractMPS
    AL::PeriodicVector{A}
    AR::PeriodicVector{A}
    CR::PeriodicVector{B}
    AC::PeriodicVector{A}
    function InfiniteMPS{A,B}(AL::PeriodicVector{A},
                              AR::PeriodicVector{A},
                              CR::PeriodicVector{B},
                              AC::PeriodicVector{A}=AL .* CR) where {A<:GenericMPSTensor,
                                                                     B<:MPSBondTensor}
        # verify lengths are compatible
        L = length(AL)
        L == length(AR) == length(CR) == length(AC) ||
            throw(ArgumentError("incompatible lengths of AL, AR, CR, and AC"))
        # verify tensors are compatible
        spacetype(A) == spacetype(B) ||
            throw(SpaceMismatch("incompatible space types of AL and CR"))

        return new{A,B}(AL, AR, CR, AC)
    end
    function InfiniteMPS(AL::PeriodicVector{A},
                         AR::PeriodicVector{A},
                         CR::PeriodicVector{B},
                         AC::PeriodicVector{A}=AL .* CR) where {A<:GenericMPSTensor,
                                                                B<:MPSBondTensor}
        # verify lengths are compatible
        L = length(AL)
        L == length(AR) == length(CR) == length(AC) ||
            throw(ArgumentError("incompatible lengths of AL, AR, CR, and AC"))
        # verify tensors are compatible
        spacetype(A) == spacetype(B) ||
            throw(SpaceMismatch("incompatible space types of AL and CR"))

        for i in 1:L
            N = numind(AL[i])
            N == numind(AR[i]) == numind(AC[i]) ||
                throw(SpaceMismatch("incompatible spaces at site $i"))

            # verify that the physical spaces are compatible
            phys_ind = 2:(N - 1)
            all(space.(Ref(AL[i]), phys_ind) .== space.(Ref(AR[i]), phys_ind) .==
                space.(Ref(AC[i]), phys_ind)) ||
                throw(SpaceMismatch("incompatible physical spaces at site $i"))

            # verify that the virtual spaces are compatible
            space(AL[i], 1) == dual(space(AL[i - 1], N)) &&
                space(AR[i], 1) == dual(space(AR[i - 1], N)) &&
                space(AC[i], 1) == space(AL[i], 1) &&
                space(AC[i], N) == space(AR[i], N) &&
                space(CR[i], 1) == dual(space(AL[i], N)) &&
                space(AR[i], 1) == dual(space(CR[i - 1], 2)) ||
                throw(SpaceMismatch("incompatible virtual spaces at site $i"))
            # verify that the spaces are non-zero
            dim(space(AL[i])) > 0 && dim(space(CR[i])) > 0 ||
                @warn "no fusion channels available at site $i"
        end
        return new{A,B}(AL, AR, CR, AC)
    end
end

#===========================================================================================
Constructors
===========================================================================================#

# From tensors
function InfiniteMPS(AL::AbstractVector{A}, AR::AbstractVector{A}, CR::AbstractVector{B},
                     AC::AbstractVector{A}=AL .* CR) where {A<:GenericMPSTensor,
                                                            B<:MPSBondTensor}
    return InfiniteMPS(convert(PeriodicVector{A}, AL), convert(PeriodicVector{A}, AR),
                       convert(PeriodicVector{B}, CR), convert(PeriodicVector{A}, AC))
end

function InfiniteMPS(A::AbstractVector{TA}; order=:LR,
                     kwargs...) where {TA<:GenericMPSTensor}
    AC = PeriodicArray(A)
    AL = similar.(AC)
    AR = similar.(AL)

    CR = map(AL) do a
        return id(storagetype(a), dual(right_virtualspace(a)))
    end

    ψ = InfiniteMPS(AL, AR, CR, AC)
    gaugefix!(ψ; order, kwargs...)
    return ψ
end

# From spaces
function InfiniteMPS(pspaces::AbstractVector{S}, Dspaces::AbstractVector{S};
                     kwargs...) where {S<:IndexSpace}
    return InfiniteMPS(MPSTensor.(pspaces, circshift(Dspaces, 1), Dspaces); kwargs...)
end
function InfiniteMPS(f, elt::Type{<:Number}, pspaces::AbstractVector{S},
                     Dspaces::AbstractVector{S}; kwargs...) where {S<:IndexSpace}
    return InfiniteMPS(MPSTensor.(f, elt, pspaces, circshift(Dspaces, 1), Dspaces);
                       kwargs...)
end
InfiniteMPS(d::S, D::S) where {S<:Union{Int,<:IndexSpace}} = InfiniteMPS([d], [D])
function InfiniteMPS(f, elt::Type{<:Number}, d::S,
                     D::S) where {S<:Union{Int,<:IndexSpace}}
    return InfiniteMPS(f, elt, [d], [D])
end
function InfiniteMPS(ds::AbstractVector{Int}, Ds::AbstractVector{Int})
    return InfiniteMPS(ComplexSpace.(ds), ComplexSpace.(Ds))
end
function InfiniteMPS(f, elt::Type{<:Number}, ds::AbstractVector{Int},
                     Ds::AbstractVector{Int}, kwargs...)
    return InfiniteMPS(f, elt, ComplexSpace.(ds), ComplexSpace.(Ds); kwargs...)
end

#===========================================================================================
Utility
===========================================================================================#

Base.size(ψ::InfiniteMPS, args...) = size(ψ.AL, args...)
Base.length(ψ::InfiniteMPS) = length(ψ.AL)
Base.eltype(ψ::InfiniteMPS) = eltype(ψ.AL)
Base.copy(ψ::InfiniteMPS) = InfiniteMPS(copy.(ψ.AL), copy.(ψ.AR), copy.(ψ.CR), copy.(ψ.AC))
function Base.repeat(ψ::InfiniteMPS, i::Int)
    return InfiniteMPS(repeat(ψ.AL, i), repeat(ψ.AR, i), repeat(ψ.CR, i), repeat(ψ.AC, i))
end
function Base.similar(ψ::InfiniteMPS{A,B}) where {A,B}
    return InfiniteMPS{A,B}(similar(ψ.AL), similar(ψ.AR), similar(ψ.CR), similar(ψ.AC))
end
function Base.circshift(ψ::InfiniteMPS, n)
    return InfiniteMPS(circshift(ψ.AL, n), circshift(ψ.AR, n), circshift(ψ.CR, n),
                       circshift(ψ.AC, n))
end

site_type(::Type{<:InfiniteMPS{A}}) where {A} = A
bond_type(::Type{<:InfiniteMPS{<:Any,B}}) where {B} = B

left_virtualspace(ψ::InfiniteMPS, n::Integer) = _firstspace(ψ.CR[n])
right_virtualspace(ψ::InfiniteMPS, n::Integer) = dual(_lastspace(ψ.CR[n]))

function physicalspace(ψ::InfiniteMPS{<:GenericMPSTensor{<:Any,N}}, n::Integer) where {N}
    if N == 1
        return ProductSpace{spacetype(ψ)}()
    elseif N == 2
        return space(ψ.AL[n], 2)
    else
        return ProductSpace{spacetype(ψ),N - 1}(space.(Ref(ψ.AL[n]),
                                                       Base.front(Base.tail(TensorKit.allind(ψ.AL[n])))))
    end
end

TensorKit.space(ψ::InfiniteMPS{<:MPSTensor}, n::Integer) = space(ψ.AC[n], 2)
function TensorKit.space(ψ::InfiniteMPS{<:GenericMPSTensor}, n::Integer)
    t = ψ.AC[n]
    S = spacetype(t)
    return ProductSpace{S}(space.(Ref(t), Base.front(Base.tail(TensorKit.allind(t)))))
end

TensorKit.norm(ψ::InfiniteMPS) = norm(ψ.AC[1])
function TensorKit.normalize!(ψ::InfiniteMPS)
    normalize!.(ψ.CR)
    normalize!.(ψ.AC)
    return ψ
end

function TensorKit.dot(ψ₁::InfiniteMPS, ψ₂::InfiniteMPS; krylovdim=30)
    init = similar(ψ₁.AL[1], _firstspace(ψ₂.AL[1]) ← _firstspace(ψ₁.AL[1]))
    randomize!(init)
    vals, _, convhist = eigsolve(TransferMatrix(ψ₂.AL, ψ₁.AL), init, 1, :LM,
                                 Arnoldi(; krylovdim=krylovdim))
    convhist.converged == 0 && @warn "dot failed to converge: normres = $(convhist.normres)"
    return vals[1]
end

function Base.show(io::IO, ::MIME"text/plain", ψ::InfiniteMPS)
    L = length(ψ)
    println(io, L == 1 ? "single site" : "$L-site", " InfiniteMPS:")
    context = IOContext(io, :typeinfo => eltype(ψ), :compact => true)
    return show(context, ψ)
end
Base.show(io::IO, ψ::InfiniteMPS) = show(convert(IOContext, io), ψ)
function Base.show(io::IOContext, ψ::InfiniteMPS)
    charset = (; mid="├", ver="│", dash="──")
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    L = length(ψ)
    println(io, charset.ver, "   ⋮")
    for site in reverse(1:L)
        if site < half_screen_rows || site > L - half_screen_rows
            if site == L
                println(io, charset.ver, " CR[$site]: ", ψ.CR[site])
            end
            println(io, charset.mid, charset.dash, " AL[$site]: ", ψ.AL[site])
        elseif site == half_screen_rows
            println(io, charset.ver, "⋮")
        end
    end
    println(io, charset.ver, "   ⋮")
    return nothing
end

#===========================================================================================
Fixedpoints
===========================================================================================#

"""
    l_RR(ψ, location)

Left dominant eigenvector of the `AR`-`AR` transfermatrix.
"""
l_RR(ψ::InfiniteMPS, loc::Int=1) = adjoint(ψ.CR[loc - 1]) * ψ.CR[loc - 1]

"""
    l_RL(ψ, location)

Left dominant eigenvector of the `AR`-`AL` transfermatrix.
"""
l_RL(ψ::InfiniteMPS, loc::Int=1) = ψ.CR[loc - 1]

"""
    l_LR(ψ, location)

Left dominant eigenvector of the `AL`-`AR` transfermatrix.
"""
l_LR(ψ::InfiniteMPS, loc::Int=1) = ψ.CR[loc - 1]'

"""
    l_LL(ψ, location)

Left dominant eigenvector of the `AL`-`AL` transfermatrix.
"""
function l_LL(ψ::InfiniteMPS{A}, loc::Int=1) where {A}
    return isomorphism(storagetype(A), space(ψ.AL[loc], 1), space(ψ.AL[loc], 1))
end

"""
    r_RR(ψ, location)

Right dominant eigenvector of the `AR`-`AR` transfermatrix.
"""
function r_RR(ψ::InfiniteMPS{A}, loc::Int=length(ψ)) where {A}
    return isomorphism(storagetype(A), domain(ψ.AR[loc]), domain(ψ.AR[loc]))
end

"""
    r_RL(ψ, location)

Right dominant eigenvector of the `AR`-`AL` transfermatrix.
"""
r_RL(ψ::InfiniteMPS, loc::Int=length(ψ)) = ψ.CR[loc]'

"""
    r_LR(ψ, location)

Right dominant eigenvector of the `AL`-`AR` transfermatrix.
"""
r_LR(ψ::InfiniteMPS, loc::Int=length(ψ)) = ψ.CR[loc]

"""
    r_LL(ψ, location)

Right dominant eigenvector of the `AL`-`AL` transfermatrix.
"""
r_LL(ψ::InfiniteMPS, loc::Int=length(ψ)) = ψ.CR[loc] * adjoint(ψ.CR[loc])
