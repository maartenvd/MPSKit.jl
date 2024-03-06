"
    Window(left,middle,right)

    general struct of an object with a left, middle and right part.
"
struct Window{L,M,R}
    left::L
    middle::M
    right::R
end

function Base.:+(window1::Window, window2::Window)
    return Window(window1.left + window2.left, window1.middle + window2.middle,
                  window1.right + window2.right)
end

# Holy traits
TimeDependence(x::Window) = istimed(x) ? TimeDependent() : NotTimeDependent()
istimed(x::Window) = istimed(x.left) || istimed(x.middle) || istimed(x.right)

function _eval_at(x::Window, args...)
    return Window(_eval_at(x.left, args...), _eval_at(x.middle, args...),
                  _eval_at(x.right, args...))
end
safe_eval(::TimeDependent, x::Window, t::Number) = _eval_at(x, t)

# For users
(x::Window)(t::Number) = safe_eval(x, t)