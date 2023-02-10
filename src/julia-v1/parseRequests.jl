const DROPOFF_ALPHA = 1.5
# Constants for reading data line
const IDX = 1
const SRC_POINT_X = 2
const SRC_POINT_Y = 3
const DST_POINT_X = 4
const DST_POINT_Y = 5
const DRT = 6
const MRT= 7
const PICKUP_OR_DROPOFF_TIME = 8
const ISPICK_TIME = 9

struct Point
    x::Int32
    y::Int32
    Point(x, y) = new(x, y)
end

struct Request
    load::Int32 # number of passengers for this request
    id::Int32
    src::Point
    dst::Point
    direct_ride_time::Float64
    max_ride_time::Float64
    pickup_time::Float64
    dropoff_time::Float64
    is_pickup::Bool
end

function request_from_dataline(line::AbstractString)
    parts = split(line, "\t")
    parts = filter(part -> part != "", parts)
    # the parts should ateleast of size 9
    if length(parts) < 9
        return nothing
    end

    time = parse(Float64, parts[PICKUP_OR_DROPOFF_TIME])
    drt = parse(Float64, parts[DRT])
    mrt = parse(Float64, parts[MRT])
    return Request(1,
                    parse(Int32, parts[IDX]),
                    Point(parse(Int32, parts[SRC_POINT_X]), parse(Int32, parts[SRC_POINT_Y])),
                    Point(parse(Int32, parts[DST_POINT_X]), parse(Int32, parts[DST_POINT_Y])),
                    drt,
                    mrt,
                    time,
                    time + (DROPOFF_ALPHA * drt),
                    Bool(parse(Int8, parts[ISPICK_TIME]))
                    )
end


function parseData(noofCustomers::Int64,
    serviceDuration::Int64, areaOfService::Int64)

    basepath = "/Users/raghuveernaraharisetti/mscs/dail-a-ride/Dail-A-Ride-TabuSearch-CPU"
    filepath = "$(basepath)/DARPDATASET/Temportal-DS/nCustomers_$(noofCustomers)/Temporal_SD$(serviceDuration)hrs_SA$(areaOfService)km.txt"
    print(filepath)
    requests = Request[]
    filelines = readlines(filepath)

    for line in filelines
        req = request_from_dataline(line)
        if isnothing(req)
            println("nothing")
            continue
        end
        push!(requests, req)
    end

    return requests
end

