include("darp.jl")
include("parseRequests.jl")
const RVAL = Dict{Int,Int}

# excepts each routes start and end depot to be actual start and deport nodes
function route_values(route, darp)
    A = RVAL([0 => 0])
    w = RVAL([0 => 0])
    B = RVAL([0 => 0])
    D = RVAL([0 => 0])
    y = RVAL([0 => 0])

    for (prev, i) in zip(route, route[2:end])
        A[i] = D[prev] + darp.travel_time(prev, i)
        B[i] = A[i]
        D[i] = B[i] + darp.d[i]
        w[i] = B[i] - A[i]
        y[i] = y[prev] + darp.q[i]
    end

    return A, w, B, D, y
end

function calc_optimization_val(darp::DARP, routes::Route)
    rvalues = Dict(k => route_values(routes[k], darp) for k in keys(routes))
    function c_of_route(route::Array{Int64})
        cost = 0
        for (prev, i) in zip(route, route[1])
            cost += darp.time_travel(prev, i)
        end
        return cost
    end # c_of_route end
    c = sum(c_of_route(route) for route in routes)

    function max_cap_in_route(k::Int64)
        return max(rvalues[k][5].values())
    end
    q = sum(max(max_cap_in_route(k) - darp.Q, 0) for k in routes.keys())

    function duration_of_route(k::Int64)
        return rvalues[k][4][darp.end_depot]
    end
    d = sum(max(duration_of_route(k) - darp.T_route, 0) for k in routes.keys())

    function late_quantity(k, i)
        _, li_pickup = darp.tw[i]
        _, li_dropoff = darp.tw[-i]
        Bi_pickup = rvalues[k][3][i]
        Bi_dropoff = rvalues[k][3][-i]
        return max(Bi_pickup - li_pickup, 0) + max(Bi_dropoff - li_dropoff, 0)
    end
    route_requests = Dict(k => Set([abs(i) for i in routes[k]]))
    route_late = Dict(k => sum(late_quantity(k, i) for i in route_requests(k)))
    w = sum(route_late.values())
    return c + q + d + w
end