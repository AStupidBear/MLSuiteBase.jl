__precompile__(true)

module MLSuiteBase

using JSON, SHA, Glob
import ScikitLearnBase: is_classifier

export @grid, gridparams, reset!, istree, isrnn, is_ranker, support_multiclass, signone, modelhash, available_memory

reset!(m) = nothing
istree(m) = false
isrnn(m) = false
support_multiclass(m) = is_classifier(m)
is_ranker(m) = false
modelhash(m) = hash(m)

gridparams(grid, combine = Iterators.product) =
    vec([merge(v...) for v in combine(gridparams.(grid)...)])

gridparams(pair::Pair) = [Dict(pair[1] => v) for v in pair[2]]

gridparams(grid::Vector{<:Vector}) = mapreduce(gridparams, vcat, grid)

Base.sign(x::Real, Θ) = ifelse(x < -Θ, oftype(x, -1), ifelse(x > Θ, one(x), zero(x)))

function signone(x::Real, Θ = zero(x))
    y = sign(x, Θ)
    !iszero(y) && return y
    rand([-one(x), one(x)])
end

function embed_include(file, mod)
    if isfile(file)
        lines = readlines(file)
    else
        lines = split(file, '\n')
    end
    for (n, line) in enumerate(lines)
        startswith(line, '#') && continue
        m = match(r"(?<=include\().*?(?=\))", line)
        isnothing(m) && continue
        src = mod.eval(Meta.parse(m.match))
        src = isabspath(src) ? src : joinpath(dirname(file), src)
        lines[n] = embed_include(src, mod)
    end
    join(lines, '\n')
end

macro grid(n, ex)
    params = __module__.eval(ex)
    file = string(__source__.file)
    if !isfile(file) || isinteractive()
        return params[1]
    end
    line = __source__.line
    lines = split(embed_include(file, __module__), '\n')
    root = "job_" * splitext(basename(file))[1]
    project = "#!/bin/bash\nexport JULIA_PROJECT=$(Base.load_path()[1])\n"
    for param in rand(params, n)
        dir = mkpath(joinpath(root, bytes2hex(sha1(json(param)))))
        open(joinpath(dir, "param.json"), "w") do io
            JSON.print(io, param, 4)
        end
        write(joinpath(dir, "julia.sh"), project, raw"""
        source ~/.bashrc
        cd $(dirname $0)
        [ -f done ] && exit 0
        julia main.jl 2>&1 | tee julia.out
        [ $? == 0 ] && touch done
        """)
        write(joinpath(dir, "mpi.sh"), project, raw"""
        cd $(dirname $0)
        [ -f done ] && exit 0
        mpirun julia main.jl 2>&1 | tee mpi.out
        [ $? == 0 ] && touch done
        """)
        write(joinpath(dir, "slurm.sh"), project, raw"""
        cd $(dirname $0)
        [ -f done ] && exit 0
        srun julia main.jl
        [ $? == 0 ] && touch done
        """)
        var_name = match(r"(.+?)(?=\s+\=)", lines[line]).match
        lines[line] = "$var_name = $param"
        write(joinpath(dir, "main.jl"), join(lines, '\n'))
        foreach(sh -> chmod(sh, 0o775), glob("*.sh", dir))
    end
    return :(exit())
end


function available_memory()
    if Sys.iswindows()
        Sys.free_memory() / 1024^3
    else
        regex = r"MemAvailable:\s+(\d+)\s+kB"
        meminfo = read("/proc/meminfo", String)
        mem = match(regex, meminfo).captures[1]
        parse(Int, mem) / 1024^2
    end
end

end