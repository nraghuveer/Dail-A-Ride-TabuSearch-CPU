import Base.Threads
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

const TabuMemory = Dict{MoveParams, Int64}

function local_search(darp::DARP, iterations::Int64,
                        N_SIZE::Int64, rawInitRoute::Route)
    tabuMem = TabuMemory()
    bestRoute::Route = deepcopy(rawInitRoute)
    bestVal::Float64 = calc_optimization_val(darp, bestRoute)
    curRoute = bestRoute
    curVal = bestVal
    for curIteration in 1:iterations
        newRoute::Route, newVal::Float64 = do_local_search!(curIteration, tabuMem, darp, curRoute, N_SIZE)
        if newVal <= bestVal
            bestRoute = deepcopy(newRoute)
            bestVal = newVal
        end
        println("$(curIteration) - $(newVal) - $(bestVal)")
        curRoute = newRoute
        curVal = newVal
    end
    return bestRoute, bestVal
end

function generate_random_moves(iterationNum::Int64, tabuMem::TabuMemory,
                                nR::Int64, nV::Int64,
                                size::Int64, routes::Route) Array{MoveParams}
    moves::Set{MoveParams} = Set([])
    vehicles = collect(nR+1:nR+nV)
    vehicleWeights = Weights(fill(1, nV))
    seed = 0
    while length(moves) < size
        seed = seed + 1
        rng = MersenneTwister(seed)
        k1, k2 = StatsBase.sample(rng, vehicles, vehicleWeights, 2, replace=false)
        if length(routes[k1]) == 0
            continue
        end
        # pick a request from k1
        i = StatsBase.sample(rng, routes[k1], Weights(fill(1, length(routes[k1]))), 1)
        i = abs(i[1])
        len_k2::Int64 = length(routes[k2])
        if len_k2 == 0
            continue
        end
        p1, p2 = StatsBase.sample(rng, 1:len_k2, Weights(fill(1, len_k2)),
            2, replace=false, ordered=true)
        param = MoveParams(i, k1, k2, p1, p2)
        if !(param in moves) && (get(tabuMem, param, -40) <= iterationNum + 40) # TODO: Remove this constant
            push!(moves, param)
        end
    end
    return collect(moves)
end

function apply_move(routes::Route, move::MoveParams)
    newRoutes = deepcopy(routes)
    # remove "i" from k1 route
    deleteat!(newRoutes[move.k1], findall(x -> abs(x) == move.i, newRoutes[move.k1]))
    insert!(newRoutes[move.k2], move.p1, move.i)
    insert!(newRoutes[move.k2], move.p2, -move.i)
    return newRoutes
end

function do_local_search!(iterationNum::Int64, tabuMem::TabuMemory,
                            darp::DARP, routes::Route, N_SIZE::Int64)
    moves = generate_random_moves(iterationNum, tabuMem, darp.nR, darp.nV, N_SIZE, routes)
    for move in moves
        tabuMem[move] = iterationNum
    end
    scores = fill(floatmin(Float64), N_SIZE)
    Threads.@threads for tid in 1:N_SIZE
        newRoutes = apply_move(routes, moves[tid])
        scores[tid] = calc_optimization_val(darp, newRoutes)
    end
    minScore, idx = findmin(scores)
    return apply_move(routes, moves[idx]), minScore
end
