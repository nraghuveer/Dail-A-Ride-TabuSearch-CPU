import Threads
import StatsBase
import Random
include("optimization_fn.jl")
include("darp.jl")


struct MoveParams
    i::Int64
    k1::Int64
    k2::Int64
    p1::Int64
    p2::Int64
end

function local_search(darp::DARP, ITERATIONS::Int64,
                N_SIZE::Int64, rawInitRoute::Route)
    bestRoute = rawInitRoute
    bestVal = calc_optimization_val(darp, bestRoute)
    curRoute = bestRoute
    curVal = bestVal
    for curIteration in 1::ITERATIONS
        newRoute = do_local_search(darp, curRoute, N_SIZE)
        newVal = calc_optimization_val(darp, newRoute)
        if newVal <= bestVal
            bestRoute = newRoute
            bestVal = newVal
        end
        println("${curIteration} - ${newVal}//${bestVal}")
        curRoute = newRoute
        curVal = newVal
    end
    return bestRoute, bestVal
end

function generate_random_moves(nR::Int64, nV::Int64,
                    size::Int64, routes::Route) Set{MoveParams}
    moves::Set{MoveParams} = Set([])
    vehicles = collect(nR+1:nR+nV)
    vehicleWeights = Weights(fill(1, nV))
    seed = 0
    while length(moves) < size
        seed = seed + 1
        rng = MersenneTwister(seed)
        k1, k2 = StatsBase.sample(rng, vehicles, vehicleWeights, 2)
        if length(routes[k1]) == 0
            continue
        end
        # pick a request from k1
        i = StatsBase.sample(rng, routes[k1], Weights(fill(1, length(routes[k1]))), 1)
        i = abs(i)
        len_k2 = length(routes[k2])
        p1, p2 = StatsBase.sample(rng, 1::len_k2+2, Weights(fill(1, len_k2+1)), 2, ordered=true)
        param = MoveParams(i, k1, k2, p1, p2)
        if param not in moves
            push!(moves, param)
            break
        end
    end
    return moves
end

function apply_move(routes::Route, move::MoveParams)
    newRoutes = deepcopy(routes)
    # remove "i" from k1 route
    deleteat!(newRoutes[k1], findall(x -> abs(x) == move.i, newRoutes[k1]))
    insert!(newRoutes[k2], move.p1, move.i)
    insert!(newRoutes[k2], move.p2, -move.i)
    return routes
end

function do_local_search(darp::DAPR, routes::Route, N_SIZE::Int64)
    moves = generate_random_moves(darp.nR, darp.nV, N_SIZE, routes)
    scores = fill(floatmin.Float64, N_SIZE)
    Threads.@thread for tid in 1::N_SIZE
        newRoutes = apply_move(routes, moves[tid])
        scores[tid] = calc_optimization_val(darp, newRoutes)
    end
    minScore, idx = findmin(scores)
    return apply_move(routes, moves[idx]), minScore
end
