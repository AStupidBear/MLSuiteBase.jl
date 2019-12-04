__precompile__(true)

module MLSuiteBase

import ScikitLearnBase: is_classifier

export paramgrid, reset!, istree, isrnn, is_ranker, support_multiclass, signone, modelhash

reset!(m) = nothing
istree(m) = false
isrnn(m) = false
support_multiclass(m) = is_classifier(m)
is_ranker(m) = false
modelhash(m) = hash(m)

paramgrid(grid::AbstractDict, combine = Iterators.product) =
    [Dict(zip(keys(grid), v)) for v in combine(values(grid)...)]

Base.sign(x::Real, Θ) = ifelse(x < -Θ, oftype(x, -1), ifelse(x > Θ, one(x), zero(x)))

function signone(x::Real, Θ = zero(x))
    y = sign(x, Θ)
    !iszero(y) && return y
    rand([-one(x), one(x)])
end

end