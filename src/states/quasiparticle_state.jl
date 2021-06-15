#=
Should not be constructed by the user - acts like a vector (used in eigsolve)
I think it makes sense to see these things as an actual state instead of return an array of B tensors (what we used to do)
This will allow us to plot energy density (finite qp) and measure observeables.
=#

struct LeftGaugedQP{S,T1,T2}
    # !(left_gs === right_gs) => domain wall excitation
    left_gs::S
    right_gs::S

    VLs::Vector{T1} # AL' VL = 0 (and VL*X = B)
    Xs::Vector{T2} # contains variational parameters

    momentum::Float64
end

struct RightGaugedQP{S,T1,T2}
    # !(left_gs === right_gs) => domain wall excitation
    left_gs::S
    right_gs::S

    Xs::Vector{T2}
    VRs::Vector{T1}

    momentum::Float64
end

#constructors
function LeftGaugedQP(datfun,left_gs,right_gs=left_gs;sector = first(sectors(oneunit(virtualspace(left_gs,1)))),momentum=0.0)
    #find the left null spaces for the TNS
    excitation_space = ℂ[typeof(sector)](sector => 1);
    VLs = [adjoint(rightnull(adjoint(v))) for v in left_gs.AL]
    Xs = [TensorMap(datfun,eltype(left_gs.AL[1]),space(VLs[loc],3)',excitation_space'*virtualspace(right_gs,loc)) for loc in 1:length(left_gs)]
    left_gs isa InfiniteMPS || momentum == zero(momentum) || @warn "momentum is ignored for finite quasiparticles"
    LeftGaugedQP(left_gs,right_gs,VLs,Xs,momentum)
end

function RightGaugedQP(datfun,left_gs,right_gs=left_gs;sector = first(sectors(oneunit(virtualspace(left_gs,1)))),momentum=0.0)
    #find the left null spaces for the TNS
    excitation_space = ℂ[typeof(sector)](sector => 1);
    VRs = [adjoint(leftnull(adjoint(v))) for v in _permute_tail.(left_gs.AR)]
    Xs = [TensorMap(datfun,eltype(left_gs.AL[1]),virtualspace(right_gs,loc-1)',excitation_space'*space(VRs[loc],1)) for loc in 1:length(left_gs)]
    left_gs isa InfiniteMPS || momentum == zero(momentum) || @warn "momentum is ignored for finite quasiparticles"
    RightGaugedQP(left_gs,right_gs,Xs,VRs,momentum)
end

#gauge dependent code
Base.similar(v::LeftGaugedQP,t=eltype(v)) = LeftGaugedQP(v.left_gs,v.right_gs,v.VLs,map(e->similar(e,t),v.Xs),v.momentum)
Base.similar(v::RightGaugedQP,t=eltype(v)) = RightGaugedQP(v.left_gs,v.right_gs,map(e->similar(e,t),v.Xs),v.VRs,v.momentum)

Base.getindex(v::LeftGaugedQP,i::Int) = v.VLs[mod1(i,end)]*v.Xs[mod1(i,end)];
Base.getindex(v::RightGaugedQP,i::Int) = @tensor t[-1 -2;-3 -4] := v.Xs[mod1(i,end)][-1,-3,1]*v.VRs[mod1(i,end)][1,-2,-4];

function Base.setindex!(v::LeftGaugedQP,B,i::Int)
    v.Xs[mod1(i,end)] = v.VLs[mod1(i,end)]'*B
    v
end
function Base.setindex!(v::RightGaugedQP,B,i::Int)
    @tensor v.Xs[mod1(i,end)][-1; -2 -3]:=B[-1,1,-2,2]*conj(v.VRs[mod1(i,end)][-3,1,2])
    v
end

#conversion between gauges (partially implemented)
function Base.convert(::Type{RightGaugedQP},input::LeftGaugedQP{S}) where S<:InfiniteMPS
    rg = RightGaugedQP(zeros,input.left_gs,input.right_gs,sector = first(sectors(utilleg(input))), momentum = input.momentum);
    len = length(input);

    #construct environments
    rBs = [@tensor t[-1 -2;-3] := input[len][-1,2,-2,3]*conj(input.right_gs.AR[len][-3,2,3])*exp(1im*input.momentum)]
    for i in len-1:-1:1
        t = exci_transfer_right(rBs[end],input.left_gs.AL[i],input.right_gs.AR[i]);
        @tensor t[-1 -2;-3] += input[i][-1,2,-2,3]*conj(input.right_gs.AR[i][-3,2,3])
        push!(rBs,exp(1im*input.momentum)*t);
    end
    rBs = reverse(rBs);

    (rBE,convhist) = @closure linsolve(rBs[1],rBs[1],GMRES()) do x
        y = transfer_right(x,input.left_gs.AL,input.right_gs.AR)*exp(1im*input.momentum*len)
        if input.trivial
            @tensor y[-1 -2;-3]-=y[1,-2,2]*l_LR(input.right_gs)[2,1]*r_LR(input.right_gs)[-1,-3]
        end
        x-y
    end
    convhist.converged == 0 && @warn "failed to converge $(convhist.normres)"

    rBs[1] = rBE;
    for i in len:-1:2
        rBE = transfer_right(rBE,input.left_gs.AL[i],input.right_gs.AR[i])*exp(1im*input.momentum);
        rBs[i] += rBE;
    end

    #final contraction is now easy
    for i in 1:len
        @tensor T[-1 -2;-3 -4] := input.left_gs.AL[i][-1,-2,1]*rBs[mod1(i+1,end)][1,-3,-4]
        @tensor T[-1 -2;-3 -4] += input[i][-1 -2;-3 -4]
        rg[i] = T
    end

    rg
end
function Base.convert(::Type{LeftGaugedQP},input::RightGaugedQP{S}) where S<:InfiniteMPS
    lg = LeftGaugedQP(zeros,input.left_gs,input.right_gs,sector = first(sectors(utilleg(input))), momentum = input.momentum);
    len = length(input);

    lBs = [@tensor t[-1 -2;-3] := input[1][1,2,-2,-3]*conj(input.left_gs.AL[1][1,2,-1])*exp(-1im*input.momentum)];
    for i in 2:len
        t = exci_transfer_left(lBs[end],input.right_gs.AR[i],input.left_gs.AL[i]);
        @tensor t[-1 -2;-3] += input[i][1,2,-2,-3]*conj(input.left_gs.AL[i][1,2,-1])
        push!(lBs,t*exp(-1im*input.momentum));
    end

    (lBE,convhist) = @closure linsolve(lBs[end],lBs[end],GMRES()) do x
        y = transfer_left(x,input.right_gs.AR,input.left_gs.AL)*exp(-1im*input.momentum*len)
        if input.trivial
            @tensor y[-1 -2;-3] -= y[1,-2,2]*r_RL(input.right_gs)[2,1]*l_RL(input.right_gs)[-1,-3]
        end
        x-y
    end
    convhist.converged == 0 && @warn "failed to converge $(convhist.normres)"

    lBs[end] = lBE;
    for i in 1:len-1
        lBE = transfer_left(lBE,input.right_gs.AR[i],input.left_gs.AL[i])*exp(-1im*input.momentum)
        lBs[i]+=lBE;
    end


    for i in 1:len
        @tensor T[-1 -2;-3 -4] := lBs[mod1(i-1,len)][-1,-3,1]*input.right_gs.AR[i][1,-2,-4]
        @tensor T[-1 -2;-3 -4] += input[i][-1 -2;-3 -4]
        lg[i] = T
    end

    lg
end

# gauge independent code
const QP{S,T1,T2} = Union{LeftGaugedQP{S,T1,T2},RightGaugedQP{S,T1,T2}} where {S,T1,T2};
const FiniteQP{S,T1,T2} = QP{S,T1,T2} where {S<:FiniteMPS}
const InfiniteQP{S,T1,T2} = QP{S,T1,T2} where {S<:InfiniteMPS}

utilleg(v::QP) = space(v.Xs[1],2)
Base.copy(a::QP) = copy!(similar(a),a)
Base.copyto!(a::QP,b::QP) = copy!(a,b);
function Base.copy!(a::T,b::T) where T<:QP
    for (i,j) in zip(a.Xs,b.Xs)
        copy!(i,j)
    end
    a
end
function Base.getproperty(v::QP,s::Symbol)
    if s == :trivial
        return v.left_gs === v.right_gs
    else
        return getfield(v,s)
    end
end

function Base.:-(v::T,w::T) where T<:QP
    t = similar(v)
    t.Xs[:] = (v.Xs-w.Xs)[:]
    t
end
function Base.:+(v::T,w::T) where T<:QP
    t = similar(v)
    t.Xs[:] = (v.Xs+w.Xs)[:]
    t
end
LinearAlgebra.dot(v::T, w::T)  where T<:QP = sum(dot.(v.Xs, w.Xs))
LinearAlgebra.norm(v::QP) = norm(norm.(v.Xs))
LinearAlgebra.normalize!(w::QP) = rmul!(w,1/norm(w));
Base.length(v::QP) = length(v.Xs)
Base.eltype(v::QP) = eltype(eltype(v.Xs)) # - again debateable, need scaltype

function LinearAlgebra.mul!(w::T, a, v::T)  where T<:QP
    @inbounds for (i,j) in zip(w.Xs,v.Xs)
        LinearAlgebra.mul!(i, a, j)
    end
    return w
end

function LinearAlgebra.mul!(w::T, v::T, a)  where T<:QP
    @inbounds for (i,j) in zip(w.Xs,v.Xs)
        LinearAlgebra.mul!(i, j, a)
    end
    return w
end
function LinearAlgebra.rmul!(v::QP, a)
    for x in v.Xs
        LinearAlgebra.rmul!(x, a)
    end
    return v
end

function LinearAlgebra.axpy!(a, v::T, w::T)  where T<:QP
    @inbounds for (i,j) in zip(w.Xs,v.Xs)
        LinearAlgebra.axpy!(a, j, i)
    end
    return w
end
function LinearAlgebra.axpby!(a, v::T, b, w::T)  where T<:QP
    @inbounds for (i,j) in zip(w.Xs,v.Xs)
        LinearAlgebra.axpby!(a, j, b, i)
    end
    return w
end

Base.:*(v::QP, a) = mul!(similar(v),a,v)
Base.:*(a, v::QP) = mul!(similar(v),a,v)

Base.zero(v::QP) = v*0;

function Base.convert(::Type{<:FiniteMPS},v::QP{S}) where S <: FiniteMPS
    #very slow and clunky, but shouldn't be performance critical anyway

    elt = eltype(v)

    utl = utilleg(v); ou = oneunit(utl); utsp = ou ⊕ ou;
    upper = isometry(Matrix{elt},utsp,ou); lower = leftnull(upper);
    upper_I = upper*upper'; lower_I = lower*lower'; uplow_I = upper*lower';

    Ls = v.left_gs.AL[1:end];
    Rs = v.right_gs.AR[1:end];

    #step 0 : fuse the utility leg of B with the first leg of B
    orig_Bs = map(i->v[i],1:length(v))
    Bs = @closure map(orig_Bs) do t
        frontmap = isomorphism(storagetype(t),fuse(utl*_firstspace(t)),utl*_firstspace(t));
        @tensor tt[-1 -2;-3]:=t[1,-2,2,-3]*frontmap[-1,2,1]
    end

    function simplefuse(temp)
        frontmap = isomorphism(storagetype(temp),fuse(space(temp,1)*space(temp,2)),space(temp,1)*space(temp,2))
        backmap = isomorphism(storagetype(temp),space(temp,5)'*space(temp,4)',fuse(space(temp,5)'*space(temp,4)'))

        @tensor tempp[-1 -2;-3] := frontmap[-1,1,2]*temp[1,2,-2,3,4]*backmap[4,3,-3]
    end

    #step 1 : pass utl through Ls
    passer = isomorphism(Matrix{elt},utl,utl);
    for (i,L) in enumerate(Ls)
        @tensor temp[-1 -2 -3 -4;-5]:=L[-2,-3,-4]*passer[-1,-5]
        Ls[i] = simplefuse(temp)
    end

    #step 2 : embed all Ls/Bs/Rs in the same space
    superspaces = map(zip(Ls,Rs)) do (L,R)
        supremum(space(L,1),space(R,1))
    end
    push!(superspaces,supremum(_lastspace(Ls[end])',_lastspace(Rs[end])'))

    for i in 1:(length(v)+1)
        Lf = isometry(Matrix{elt},superspaces[i],i <= length(v) ? _firstspace(Ls[i]) : _lastspace(Ls[i-1])')
        Rf = isometry(Matrix{elt},superspaces[i],i <= length(v) ? _firstspace(Rs[i]) : _lastspace(Rs[i-1])')

        if i <= length(v)
            @tensor Ls[i][-1 -2;-3] := Lf[-1,1]*Ls[i][1,-2,-3]
            @tensor Rs[i][-1 -2;-3] := Rf[-1,1]*Rs[i][1,-2,-3]
            @tensor Bs[i][-1 -2;-3] := Lf[-1,1]*Bs[i][1,-2,-3]
        end

        if i>1
            @tensor Ls[i-1][-1 -2;-3] := Ls[i-1][-1 -2;1]*conj(Lf[-3,1])
            @tensor Rs[i-1][-1 -2;-3] := Rs[i-1][-1 -2;1]*conj(Rf[-3,1])
            @tensor Bs[i-1][-1 -2;-3] := Bs[i-1][-1 -2;1]*conj(Rf[-3,1])
        end
    end

    #step 3 : fuse the correct *_I with the correct tensor (and enforce boundary conditions)
    function doboundary(temp1,pos)
        if pos == 1
            @tensor temp2[-1 -2 -3 -4;-5] := temp1[1,-2,-3,-4,-5]*conj(upper[1,-1])
        elseif pos == length(v)
            @tensor temp2[-1 -2 -3 -4;-5] := temp1[-1 -2 -3 -4 1]*lower[1,-5]
        else
            temp2 = temp1;
        end

        temp2
    end

    for i in 1:length(v)
        @tensor temp[-1 -2 -3 -4; -5] := Ls[i][-2,-3,-4]*upper_I[-1,-5]
        temp = doboundary(temp,i);
        Ls[i] = simplefuse(temp) * (i<length(v));

        @tensor temp[-1 -2 -3 -4; -5] := Rs[i][-2,-3,-4]*lower_I[-1,-5]
        temp = doboundary(temp,i);
        Rs[i] = simplefuse(temp) * (i>1);

        @tensor temp[-1 -2 -3 -4; -5] := Bs[i][-2,-3,-4]*uplow_I[-1,-5]
        temp = doboundary(temp,i);
        Bs[i] = simplefuse(temp);
    end

    return FiniteMPS(Ls+Rs+Bs,normalize=false)
end
