__precompile__(true)

module MLSuiteBase

import ScikitLearnBase: is_classifier

export paramgrid, reset!, istree, isrnn, is_ranker, support_multiclass

reset!(m) = nothing
istree(m) = false
isrnn(m) = false
support_multiclass(m) = is_classifier(m)
is_ranker(m) = false

paramgrid(grid::AbstractDict, combine = Iterators.product) =
    [Dict(zip(keys(grid), v)) for v in combine(values(grid)...)]

end