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
    direct_ride_time::Float32
    max_ride_time::Float32
    time::Float32
    is_pickup::Bool
end

function request_from_dataline(line::AbstractString)
    parts = split(line, "\t")
    parts = filter(part -> part != "", parts)
    # the parts should ateleast of size 9
    if length(parts) < 9
        return nothing
    end

    return Request(1,
                    parse(Int32, parts[IDX]),
                    Point(parse(Int32, parts[SRC_POINT_X]), parse(Int32, parts[SRC_POINT_Y])),
                    Point(parse(Int32, parts[DST_POINT_X]), parse(Int32, parts[DST_POINT_Y])),
                    parse(Float32, parts[DRT]),
                    parse(Float32, parts[MRT]),
                    parse(Float32, parts[PICKUP_OR_DROPOFF_TIME]),
                    Bool(parse(Int8, parts[ISPICK_TIME]))
                    )
end

function parseData(noofCustomers, serviceTime, areaOfService)
    basepath = "/Users/raghuveernaraharisetti/mscs/dail-a-ride/Dail-A-Ride-TabuSearch-CPU"
    filepath = "$(basepath)/DARPDATASET/Temportal-DS/nCustomers_$(noofCustomers)/Temporal_SD$(serviceTime)hrs_SA$(areaOfService)km.txt"
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


function main()
    println(parseData(50, 2, 10))
end
main()