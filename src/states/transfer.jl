#transfer
transfer_left(v::MPSBondTensor, A::GenericMPSTensor, Ab::GenericMPSTensor=A) =
    _transpose_as(_transpose_front(Ab)' * _transpose_front(_transpose_front(v)*_transpose_tail(A)), v)
transfer_right(v::MPSBondTensor, A::GenericMPSTensor, Ab::GenericMPSTensor=A) =
    _transpose_as(_transpose_tail(_transpose_front(A)*_transpose_front(v)) * _transpose_tail(Ab)', v)


#transfer for 2 mpo tensors
transfer_left(v::MPSBondTensor,A::MPOTensor,B::MPOTensor) = @plansor t[-1;-2] := v[1;2]*A[2 3;4 -2]*conj(B[1 3;4 -1])
transfer_right(v::MPSBondTensor,A::MPOTensor,B::MPOTensor) = @plansor t[-1;-2] := A[-1 3;4 1]*conj(B[-2 3;4 2])*v[1;2]

#transfer, but there are utility legs in the middle that are passed through
transfer_left(v::AbstractTensorMap{S,N1,N2},A::GenericMPSTensor{S,N},Ab::GenericMPSTensor{S,N}=A) where {S,N1,N2,N} =
    _permute_as(Ab'*permute(_permute_front(v)*_permute_tail(A),tuple(1,ntuple(x->N1+N2-1+x,N-1)...),tuple(ntuple(x->x+1,N1+N2-2)...,N1+N2+N-1)),v)

transfer_right(v::AbstractTensorMap{S,N1,N2},A::GenericMPSTensor{S,N},Ab::GenericMPSTensor{S,N}=A) where {S,N1,N2,N} =
    _permute_as(permute(A*_permute_tail(v),tuple(1,ntuple(x->N+x,N1+N2-2)...),tuple(ntuple(x->x+1,N-1)...,N1+N2+N-1))*_permute_tail(Ab)',v)

#mpo transfer
transfer_left(v::MPSTensor,O::MPOTensor,A::MPSTensor,Ab::MPSTensor) = @plansor v[-1 -2;-3] := v[4 2;1]*A[1 3;-3]*O[2 5;3 -2]*conj(Ab[4 5;-1])
transfer_right(v::MPSTensor,O::MPOTensor,A::MPSTensor,Ab::MPSTensor) = @plansor v[-1 -2;-3] := A[-1 4;5]*O[-2 2;4 3]*conj(Ab[-3 2;1])*v[5 3;1]

#utility, allowing transfering with arrays
function transfer_left(v,A::AbstractArray,Ab::AbstractArray=A;rvec=nothing,lvec=nothing)
    for (a,b) in zip(A,Ab)
        v = transfer_left(v,a,b)
    end

    if rvec != nothing && lvec != nothing
        if v isa MPSBondTensor #normal transfer
            v -= (@plansor C[]:=rvec[1;2]*v[2;1])*lvec
        elseif v isa MPSTensor #utiity leg in the middle
            @plansor ut[-1] := v[2 -1;1]*rvec[1;2]
            v-=lvec⊗ut
        else #what have you just given me?
            @assert false
        end
    end

    return v
end
function transfer_right(v,A::AbstractArray,Ab::AbstractArray=A;rvec=nothing,lvec=nothing)
    for (a,b) in Iterators.reverse(zip(A,Ab))
        v = transfer_right(v,a,b)
    end

    if rvec != nothing && lvec != nothing
        if v isa MPSBondTensor #normal transfer
            v -= (@plansor C[] :=lvec[1;2]*v[2;1])*rvec
        elseif v isa MPSTensor #utiity leg in the middle
            @plansor ut[-1] := v[1 -1;2]*lvec[2;1]
            v-=rvec⊗ut
        else
            @assert false
        end
    end

    return v
end
transfer_left(v,O::AbstractArray,A::AbstractArray,Ab::AbstractArray) = reduce((v,x)->transfer_left(v,x[1],x[2],x[3]),zip(O,A,Ab),init=v)
transfer_right(v,O::AbstractArray,A::AbstractArray,Ab::AbstractArray) = reduce((v,x)->transfer_right(v,x[1],x[2],x[3]),Iterators.reverse(zip(O,A,Ab)),init=v)
