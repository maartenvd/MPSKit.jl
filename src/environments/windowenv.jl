"""
    WindowEnv keeps track of the environments for WindowMPS
    Changes in the infinite environments are checked and taken into account whenever the environment of the finite part is queried.

"""
struct WindowEnv{A,B,C,D} <: Cache
    window::Window{A,B,C}

    linfdeps::PeriodicArray{D} #the data we used to calculate leftenvs/rightenvs
    rinfdeps::PeriodicArray{D}
end

#automatically construct the correct leftstart/rightstart for a WindowMPS
# copying the vector where the tensors reside in makes it have another memory adress, while keeping the references of the elements
function environments(ψ::WindowMPS, O::Window, above=nothing;
                      lenvs=environments(ψ.left, O.left),
                      renvs=environments(ψ.right, O.right))
    fin_env = environments(ψ, O.middle, above, lenvs, renvs)
    return WindowEnv(Window(lenvs, fin_env, renvs), copy(ψ.left.AL), copy(ψ.right.AR))
end

function environments(ψ::WindowMPS, O::MPOHamiltonian, above, lenvs::MPOHamInfEnv,
                      renvs::MPOHamInfEnv)
    return environments(ψ, O, above, leftenv(lenvs, 1, ψ.left),
                        rightenv(renvs, length(ψ), ψ.right))
end

#Backwards compability
function environments(ψ::WindowMPS{A,B,WINDOW_FIXED,WINDOW_FIXED}, H::MPOHamiltonian,
                      above=nothing) where {A,B}
    length(H) == length(ψ.left) ||
        throw(ArgumentError("length(ψ.left) != length(H), use H=Window(Hleft,Hmiddle,Hright) instead"))
    length(H) == length(ψ.right) ||
        throw(ArgumentError("length(ψ.right) != length(H), use H=Window(Hleft,Hmiddle,Hright) instead"))
    return environments(ψ, H, above, environments(ψ.left, H), environments(ψ.right, H))
end

function environments(below::WindowMPS, above::WindowMPS)
    above.left == below.left || throw(ArgumentError("left gs differs"))
    above.right == below.right || throw(ArgumentError("right gs differs"))

    opp = fill(nothing, length(below))
    return environments(below, opp, above, l_LL(above), r_RR(above))
end

#===========================================================================================
Utility
===========================================================================================#
function Base.getproperty(ca::WindowEnv, sym::Symbol) #under review
    if sym === :left || sym === :middle || sym === :right
        return getfield(ca.window, sym)
    elseif sym === :opp #this is for derivatives. Could we remove opp field from derivatives?
        return getfield(ca.window.middle, sym)
    else
        return getfield(ca, sym)
    end
end

# when accesing the finite part of the env, use this function
function finenv(ca::WindowEnv, ψ::WindowMPS{A,B,VL,VR}) where {A,B,VL,VR}
    VL === WINDOW_FIXED || check_leftinfenv!(ca, ψ)
    VR === WINDOW_FIXED || check_rightinfenv!(ca, ψ)
    return ca.middle
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
invalidate!(ca::WindowEnv, ind) = invalidate!(ca.middle, ind)

# Check the infinite envs and recalculate the right start
function check_rightinfenv!(ca::WindowEnv, ψ::WindowMPS)
    if !all(ca.rinfdeps .=== ψ.right.AR)
        invalidate!(ca, length(ψ.middle)) #forces transfers to be recalculated lazily 

        ca.middle.rightenvs[end] = rightenv(ca.right, 0, ψ.right)
        ca.rinfdeps .= ψ.right.AR
    end
end

function check_leftinfenv!(ca::WindowEnv, ψ::WindowMPS)
    if !all(ca.linfdeps .=== ψ.left.AL)
        invalidate!(ca, 1) #forces transfers to be recalculated lazily 

        ca.middle.leftenvs[1] = leftenv(ca.left, length(ψ.left) + 1, ψ.left)
        ca.linfdeps .= ψ.left.AL
    end
end

#should be used to calculate expvals for operators over the boundary, also need views to work for this
function leftenv(ca::WindowEnv, ind, ψ::WindowMPS)
    if ind < 1
        return leftenv(ca.left, ind, ψ.left)
    elseif ind > length(ψ)
        return leftenv(ca.right, ind, ψ.right)
    else
        return leftenv(finenv(ca, ψ), ind, ψ)
    end
end

function rightenv(ca::WindowEnv, ind, ψ::WindowMPS)
    if ind < 1
        return rightenv(ca.left, ind, ψ.left)
    elseif ind > length(ψ)
        return rightenv(ca.right, ind, ψ.right)
    else
        return rightenv(finenv(ca, ψ), ind, ψ)
    end
end
