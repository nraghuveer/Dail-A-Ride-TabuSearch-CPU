include("darp.jl")
include("parseRequests.jl")

const RVAL = Dict{Int,Int}

function max_of_arr(A::Vector{Int64})
    Int64
    return reduce((x, y) -> min.(x, y), A)
end

# excepts each routes start and end depot to be actual start and deport nodes
function route_values(route::Array{Int64}, darp::DARP)
    A = RVAL([0 => 0])
    w = RVAL([0 => 0])
    B = RVAL([0 => 0])
    D = RVAL([0 => 0])
    y = RVAL([0 => 0])

    for (prev, i) in zip(route, route[2:end])
        A[i] = D[prev] + travel_time(darp, prev, i)
        B[i] = A[i]
        D[i] = B[i] + darp.d[i]
        w[i] = B[i] - A[i]
        y[i] = y[prev] + darp.q[i]
    end

    return A, w, B, D, y
end

function calc_optimization_val(darp::DARP, raw_routes::Route)
    routes::Route = Dict(k => [[darp.start_depot]; raw_routes[k]; [darp.end_depot]]
                  for k in keys(raw_routes))
    rvalues = Dict(k => route_values(
        routes[k],
        darp)
                   for k in keys(routes))
    function c_of_route(k::Int64)
        route::Array{Int64} = routes[k]
        x = [travel_time(darp, prev, i) for (prev, i) in zip(route, route[2:end])]
        return sum(x)
    end # c_of_route end
    c = sum([c_of_route(k) for k in keys(routes)])

    function max_cap_in_route(k::Int64)
        return max_of_arr(collect(values(rvalues[k][5])))
    end
    max_caps = [max(max_cap_in_route(k) - darp.Q, 0) for (k, route) in routes]
    q = sum(max_caps)

    function duration_of_route(k::Int64)
        return rvalues[k][4][darp.end_depot]
    end
    d = sum(max(duration_of_route(k) - darp.T_route, 0) for k in keys(routes))

    function late_quantity(k::Int64, i::Int64)
        _, li_pickup = darp.tw[i]
        _, li_dropoff = darp.tw[-i]
        Bi_pickup = rvalues[k][3][i]
        Bi_dropoff = rvalues[k][3][-i]
        return max(Bi_pickup - li_pickup, 0) + max(Bi_dropoff - li_dropoff, 0)
    end
    function route_requests(k::Int64)
        s = Set([abs(i) for i in routes[k]])
        return filter((i) -> i != darp.start_depot && i != darp.end_depot, s)
    end
    function total_late_for_route(k::Int64)
        return sum(late_quantity(k, i) for i in route_requests(k))
    end
    function total_late(routes::Route)
        return sum(total_late_for_route(k) for k in keys(routes))
    end
    # calc the total late for request in all the routes
    w = total_late(routes)
    println("c = $c | q = $q | d = $d | w = $w")
    return c + q + d + w
end