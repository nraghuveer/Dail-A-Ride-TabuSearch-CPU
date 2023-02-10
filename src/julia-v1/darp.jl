include("parseRequests.jl")

# TODO -> make these config driven?
const DEFAULT_SERVICE_TIME = 2
const DEFAULT_TW_OFFSET = 5 * 60 # 5 minutes in seconds
const DEFAULT_WAITTIME_AT_PICKUP = 3 * 60 # 3 minutes in seconds

struct DARP
    nR::Int32
    sd::Float64 # in seconds
    aos::Int32 # in sqmiles
    nV::Int32
    T_route::Float64 # lets say a route can run as long as service duration
    requests::Array{Request}
    start_depot::Int32
    end_depot::Int32
    Q::Int16 # vehicle capacity
    coords::Dict{Int32, Point}
    d::Dict{Int32, Int32}
    q::Dict{Int32, Int32}
    tw::Dict{Int32, Tuple{Float64, Float64}}
    w::Dict{Int32, Float64}
    function DARP(nR::Int32, sd::Int32, aos::Int32, nV::Int32, Q::Int16)
        start_depot::Int32 = 0
        end_depot::Int32 = 2*nR + 1

        sdInSeconds::Float64 = parse(Float64, sd*60*60)
        T_route::Float64 = sdInSeconds

        requests = parseData(nR, sd, aos)
        coords::Dict{int, Point} = Dict{int, Point}([])
        coords[start_depot] = requests[1].src
        coords[end_depot] = requests[1].dst

        d::Dict{Int32, Int32} = Dict{Int32, Int32}([])
        d[start_depot] = 0
        d[end_depot] = 0

        q::Dict{Int32, Int32} = Dict{Int32, Int32}([])
        q[start_depot] = 0
        q[end_depot] = 0

        tw::Dict{Int32, Tuple{Float64, Float64}} = Dict([])
        offset::Int32 = DEFAULT_TW_OFFSET
        tw[start_depot] = (parse(0, Float64), parse(offset, Float64))
        tw[end_depot] = (T_route, T_route+offset)

        w::Dict{Int32, Float64} = Dict{Int32, Float64}([])
        w[start_depot] = parse(Float64, 0)
        w[end_depot] = parse(Float64, 0)

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
        return new(nR, sdInSeconds, parse(Float64, aos*0.386102),
                    nV, T_route, requests, start_depot, end_depot,
                    Q, coords, d, q, tw, w)
    end
end

function main()
    println(parseData(50, 2, 10))
end
main()