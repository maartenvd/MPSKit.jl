# Holy traits to differentiate between time dependent and time independent types

abstract type TimeDependence end
struct TimeDependent <: TimeDependence end
struct NotTimeDependent <: TimeDependence end

TimeDependence(x) = NotTimeDependent()
istimed(::TimeDependent) = true
istimed(::NotTimeDependent) = false
istimed(x) = istimed(TimeDependence(x))

safe_eval(x, args...) = safe_eval(TimeDependence(x), x, args...)

# wrapper around _eval_at
function safe_eval(::TimeDependent, x)
    throw(ArgumentError("attempting to evaluate time-dependent object without specifiying a time"))
end
function safe_eval(::NotTimeDependent, x, t::Number)
    throw(ArgumentError("attempting to evaluate time-independent object at a time"))
end

# Internal use only, works always
_eval_at(x, args...) = x # -> this is what you should define for custom structs
