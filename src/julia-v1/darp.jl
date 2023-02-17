include("parseRequests.jl")
include("construction_kernel.jl")
using BenchmarkTools
using Base.Threads

const Route = Dict{Int64, Array{Int64}}

# TODO -> make these config driven?
const DEFAULT_SERVICE_TIME = 2
const DEFAULT_TW_OFFSET = 5 * 60 # 5 minutes in seconds
const DEFAULT_WAITTIME_AT_PICKUP = 3 * 60 # 3 minutes in seconds

struct DARP
    nR::Int64
    sd::Float64 # in seconds
    aos::Int64 # in sqmiles
    nV::Int64
    T_route::Float64 # lets say a route can run as long as service duration
    requests::Array{Request}
    start_depot::Int64
    end_depot::Int64
    Q::Int64 # vehicle capacity
    coords::Dict{Int64,Point}
    d::Dict{Int64,Int64}
    q::Dict{Int64,Int64}
    tw::Dict{Int64,Tuple{Float64,Float64}}
    w::Dict{Int64,Float64}
    function travel_time(one::Int64, two::Int64)
        pone = coords[one]
        ptwo = coords[two]
        return abs(pone.x - ptwo.x) + abs(pone.y - ptwo.y)
    end
    function DARP(nR::Int64, sd::Int64, aos::Int64, nV::Int64, Q::Int64)
        start_depot::Int64 = 0
        end_depot::Int64 = 2 * nR + 1

        sdInSeconds::Float64 = sd * 60 * 60
        aosInSqMiles::Int64 = trunc(Int64, aos * 0.386102)
        T_route::Float64 = sdInSeconds

        requests = parseData(nR, sd, aos)
        coords::Dict{Int64,Point} = Dict{Int64,Point}([])
        coords[start_depot] = requests[1].src
        coords[end_depot] = requests[1].dst

        d::Dict{Int64,Int64} = Dict{Int64,Int64}([])
        d[start_depot] = 0
        d[end_depot] = 0

        q::Dict{Int64,Int64} = Dict{Int64,Int64}([])
        q[start_depot] = 0
        q[end_depot] = 0

        tw::Dict{Int64,Tuple{Float64,Float64}} = Dict([])
        offset::Int64 = DEFAULT_TW_OFFSET
        tw[start_depot] = (0, offset)
        tw[end_depot] = (T_route, T_route + offset)

        w::Dict{Int64,Float64} = Dict{Int64,Float64}([])
        w[start_depot] = 0
        w[end_depot] = 0

        for req in requests[2:end]
            coords[req.id] = req.src
            coords[-req.id] = req.dst

            # data doesnt have specific service time at each node, so u;se const value
            d[req.id] = 2
            d[-req.id] = 2

            # change in load after each node
            q[req.id] = req.load
            q[-req.id] = -req.load

            tw[req.id] = (req.pickup_time, req.pickup_time + offset)
            tw[-req.id] = (req.dropoff_time, req.dropoff_time + offset)

            w[req.id] = DEFAULT_WAITTIME_AT_PICKUP
            w[-req.id] = 0
        end
        return new(nR, sdInSeconds, aosInSqMiles,
            nV, T_route, requests, start_depot, end_depot,
            Q, coords, d, q, tw, w)
    end
end

function main()
    darp = DARP(50, 24, 10, 5, 1)
    n = 1000 # number of tasks
    routes = fill(Route(), n)
    Threads.@threads for i in 1:n
        cur = generate(10, darp.requests, darp.nR, darp.nV)
        routes[i] = cur
    end
    println(routes[100])
end

main()