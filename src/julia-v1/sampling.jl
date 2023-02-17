# helper functions for sampling - specific to this algorithm
include("darp.jl")
using StatsBase
using Random

function generate_initial_routes(nR::Int64, nV::Int64) Dict{Int64, Array{Int64}}
    routes::Route = Dict{Int64, Array{Int64}}([])
    # the idea is to generate a route for each vehcilce
    # and also assign a vechicle to each request
    # everything should be completely random
    # so, iterate over the requests and randomly
    #  select a vehicle and assign

    # TODO: might have be little smart with this weights
    # for now, make all vehicles equal probability
    requestWeights = Weights(fill(1, nR))
    randomized_requests = StatsBase.sample(1:nR, requestWeights, nR)
    vehicles = collect(nR+1:nR+nV)
    # initialize empty routes for each vehicle
    for v in vehicles
        routes[v] = []
    end
    vehicleWeights = Weights(fill(1, nV))
    for req in randomized_requests
        k = StatsBase.sample(vehicles, vehicleWeights, 1)
        # sample generates a array, so take first item
        k = k[1]
        l = length(routes[k])
        if l === 0
            insert!(routes[k], 1, req)
        else
            p1 = rand(1:l+1)
            insert!(routes[k], p1, req)
        end
        p2 = rand(1:l+2) # +2 because we just added one in above line
        insert!(routes[k], p2, -req)
    end
    return routes
end