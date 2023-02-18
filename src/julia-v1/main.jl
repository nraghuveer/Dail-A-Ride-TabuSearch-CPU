include("darp.jl")
include("construction_kernel.jl")
include("optimization_fn.jl")
using BenchmarkTools
using Base.Threads

function main()
    darp = DARP(50, 24, 10, 5, 1)
    n = 1000 # number of tasks
    routes = fill(Route(), n)
    scores = fill(floatmin(Float64), n)
    for i in 1:n
        cur::Route = generate(10, darp.requests, darp.nR, darp.nV)
        scores[i] = calc_optimization_val(darp, cur)
        routes[i] = cur
    end
    println(routes[100])
    println(scores[105])
end

main()