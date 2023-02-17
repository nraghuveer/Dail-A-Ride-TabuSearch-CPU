# Generates initial BEST Solution
# completly random
include("parseRequests.jl")
include("sampling.jl")
include("darp.jl")

function generate(seed::Int64, requests::Array{Request}, nR::Int64, nV::Int64) Route
    initial_routes = generate_initial_routes(nR, nV)
    return initial_routes
end
