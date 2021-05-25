function _transpose_front(t::AbstractTensorMap) # make TensorMap{S,N₁+N₂-1,1}
    I1 = TensorKit.codomainind(t)
    I2 = TensorKit.domainind(t)
    transpose(t, (I1..., reverse(Base.tail(I2))...), (I2[1],))
end
function _transpose_tail(t::AbstractTensorMap) # make TensorMap{S,1,N₁+N₂-1}
    I1 = TensorKit.codomainind(t)
    I2 = TensorKit.domainind(t)
    transpose(t, (I1[1],), (I2..., reverse(Base.tail(I1))...))
end
function _transpose_as(t1::AbstractTensorMap, t2::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    I1 = (TensorKit.codomainind(t1)...,reverse(TensorKit.domainind(t1))...);

    A = ntuple(x->I1[x],N1);
    B = ntuple(x->I1[x+N1],N2);

    transpose(t1, A,B)
end

_firstspace(t::AbstractTensorMap) = space(t, 1)
_lastspace(t::AbstractTensorMap) = space(t, numind(t))

#given a hamiltonian with unit legs on the side, decompose it using svds to form a "localmpo"
function decompose_localmpo(inpmpo::AbstractTensorMap{PS,N1,N2},trunc = truncbelow(Defaults.tol)) where {PS,N1,N2}
    numind=N1+N2
    if(numind==4)
        return [transpose(inpmpo,(1,2),(3,4))]
    end

    leftind=(1,2,Int(numind/2+1))
    rightind=(ntuple(x->x+2,Val{Int((N1+N2)/2)-2}())..., ntuple(x->x+Int(numind/2+1),Val{Int((N1+N2)/2)-1}())...)

    (U,S,V) = tsvd(inpmpo,leftind,rightind,trunc = trunc)
    return [transpose(U,(1,2),(3,4));decompose_localmpo(S*V)]
end

function add_util_leg(tensor::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    ou = oneunit(_firstspace(tensor));

    util_front = isomorphism(storagetype(tensor),ou*codomain(tensor),codomain(tensor));
    util_back = isomorphism(storagetype(tensor),domain(tensor),domain(tensor)*ou);

    return util_front*tensor*util_back
end

"""
Take the L2 Tikhonov regularised inverse of a matrix `m`.

The regularisation parameter is the larger of `delta` (the optional argument that defaults
to zero) and square root of machine epsilon. The inverse is done using an SVD.
"""
function reginv(m, delta=zero(eltype(m)))
    delta = max(abs(delta), sqrt(eps(real(float(one(eltype(m)))))))
    U, S, Vdg = tsvd(m)
    Sinv = inv(real(sqrt(S^2 + delta^2*one(S))))
    minv = Vdg' * Sinv * U'
    return minv
end

function union_split(a::AbstractArray)
    T = reduce((a,b)->Union{a,b},typeof.(a))
    nA = similar(a,T);
    copy!(nA,a)
end
union_types(x::Union) = (x.a, union_types(x.b)...)
union_types(x::Type) = (x,)

function _embedders(spaces)
    totalspace = reduce(⊕,spaces);

    maps = [isometry(totalspace,first(spaces))];
    restmap = leftnull(first(maps));

    for sp in spaces[2:end]
        cm = isometry(domain(restmap),sp);

        push!(maps,restmap*cm);
        restmap = restmap*leftnull(cm);
    end

    maps
end

macro plansor(ex::Expr)
    return esc(plansor_parser(ex))
end

function plansor_parser(ex)
    t = first(TO.getinputtensorobjects(ex));

    default = TO.defaultparser(ex);
    planar = TensorKit.planar_parser(ex);

    quote
        if BraidingStyle(sectortype($t)) isa Bosonic
            $(default)
        else
            $(planar)
        end
    end
end
