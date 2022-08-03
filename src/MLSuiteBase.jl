module MLSuiteBase

using JSON, SHA, Glob
import ScikitLearnBase: is_classifier

export @grid, gridparams, reset!, istree, isrnn, is_ranker, support_multiclass, signone, modelhash, available_memory, myparam, gendir
export @gc, @staticvar, @staticdef, @redirect, logerr, @tryonce, prefix, suffix

reset!(m) = nothing
istree(m) = false
isrnn(m) = false
support_multiclass(m) = is_classifier(m)
is_ranker(m) = false
modelhash(m) = hash(m)

gridparams(grid, combine = Iterators.product) =
    unique(vec([merge(v...) for v in combine(gridparams.(reverse(grid))...)]))

gridparams(pair::Pair) = [Dict{String, Any}(pair[1] => v) for v in pair[2]]

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
    if !isfile(file) || isinteractive() ||
        get(ENV, "GRID_ENABLE", "1") == "0"
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
        write(joinpath(dir, "srun.sh"), project, raw"""
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

function myparam(params)
    if haskey(ENV, "SLURM_ARRAY_TASK_ID")
        i = parse(Int, "0" * ENV["SLURM_ARRAY_TASK_ID"]) + 1
    elseif haskey(ENV, "PBS_ARRAYID")
        i = parse(Int, "0" * ENV["PBS_ARRAYID"]) + 1
    else
        i = 1
    end
    i <= length(params) ? params[i] : exit(0)
end

function gendir(param)
    dir = mkpath(bytes2hex(sha1(json(param))))
    open(joinpath(dir, "param.json"), "w") do io
        JSON.print(io, param, 4)
    end
    return dir
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

function JSON.lower(a)
    if nfields(a) > 0
        JSON.Writer.CompositeTypeWrapper(a)
    else
        string(a)
    end
end

macro gc(exs...)
    Expr(:block, [:($ex = 0) for ex in exs]..., :(@eval GC.gc())) |> esc
end

macro staticvar(init)
    var = gensym()
    __module__.eval(:(const $var = $init))
    var = esc(var)
    quote
        global $var
        $var
    end
end

macro staticdef(ex)
    name, T = ex.args[1].args
    ref = Ref{__module__.eval(T)}()
    set = Ref(false)
    :($(esc(name)) = if $set[]
        $ref[]
    else
        $ref[] = $(esc(ex))
        $set[] = true
        $ref[]
    end)
end

macro redirect(src, ex)
    src = src == :devnull ? "/dev/null" : src
    quote
        io = open($(esc(src)), "a")
        o, e = stdout, stderr
        redirect_stdout(io)
        redirect_stderr(io)
        res = nothing
        try
            res = $(esc(ex))
            sleep(0.01)
        finally
            flush(io)
            close(io)
            redirect_stdout(o)
            redirect_stderr(e)
        end
        res
    end
end

function logerr(e, comment = "", fname = "try.err", mode = "a")
    e isa InterruptException && rethrow()
    io = IOBuffer()
    print(io, '-'^15, gethostname(), '-'^15, '\n', e)
    bt = stacktrace(catch_backtrace())
    showerror(io, e, bt)
    println(io)
    str = join(Iterators.take(String(take!(io)), 10000))
    str = join(map(l -> join(Iterators.take(l, 400)), split(str, '\n')), '\n')
    println(stdout, comment)
    println(stdout, str)
    flush(stdout)
    open(fname, mode) do fid
        println(fid, comment)
        println(fid, str)
    end
end

macro tryonce(ex)
    ex = Expr(:try, ex, :e, :(logerr(e)))
    return esc(ex)
end

prefix(str) = String(split(str, '.')[1])

suffix(str) = String(split(str, '.')[end])

end