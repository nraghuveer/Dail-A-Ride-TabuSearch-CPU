include("darp.jl")
include("construction_kernel.jl")
include("optimization_fn.jl")
include("local_search_kernel.jl")
using BenchmarkTools
using Base.Threads

# TODO: Tighten bounds for each route

function main()
    darp = DARP(500, 2, 10, 5, 1)
    n = 1000 # number of tasks
    routes = fill(Route(), n)
    scores = fill(floatmin(Float64), n)
    Threads.@threads for i in 1:n
        cur::Route = generate(10, darp.requests, darp.nR, darp.nV)
        scores[i] = calc_optimization_val(darp, cur)
        routes[i] = cur
    end
    _, idx = findmin(scores)
    local_search(darp, 1000, 20, routes[idx])
end

main()