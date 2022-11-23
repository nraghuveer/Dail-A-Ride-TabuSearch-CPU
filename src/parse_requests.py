# Growth dataset -> fixed service duration and area of service while the number of requests varies from 50 to 1000
from typing import Union
"""
SrcPoint_x= The x-coordinate of customer pick-up point.

SrcPoint_y= The y-coordinate of customer pick-up point.

DstPoint_x= The x-coordinate of customer drop-off point.

DstPoint_y= The y-coordinate of customer drop-off point.

DRT= Direct Ride Time. The time needed to transport the customer between Src and Dst point on the shortest path without any deviations.

MRT= Maximum Ride Time (calculated by adding 60 minutes to the direct ride time of the customer).

Pick-up/Drop-up time= Pickup or drop-off time of the customer determined by the flag.

Flag= A Boolean value that indicates if the time in previous column is pick-up (1) or drop-off (0) time.
"""

ALPHA = 0.8  # Alpha coefficient, used to calculate the dropoff time window
IDX = 0
SRC_POINT_X = 1
SRC_POINT_Y = 2
DST_POINT_X = 3
DST_POINT_Y = 4
DRT = 5
MRT= 6
PICKUP_OR_DROPOFF_TIME = 7
ISPICK_TIME = 8

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Request:
    def __init__(self,id, line: str, service_duration: int, area_of_service: int):

        self.service_duration = service_duration
        self.area_of_service = area_of_service

        parts = filter(lambda x: x!="", line.split("\t"))
        for i, x in enumerate(parts):
            x = x.strip("\t")
            val = int(x)
            if i == IDX:
                self.id = id
            elif i == SRC_POINT_X:
                self.src_x = val
            elif i == SRC_POINT_Y:
                self.src_y = val
            elif i == DST_POINT_X:
                self.dst_x = val
            elif i == DST_POINT_Y:
                self.dst_y = val
            elif i == DRT:
                self.drt = val
            elif i == MRT:
                self.mrt = val
            elif i == PICKUP_OR_DROPOFF_TIME:
                self.__time = val
            elif i == ISPICK_TIME:
                self.pickup_time = self.__time
                # dropoff = alpha * maxride + pickup
                self.dropoff_time = int(ALPHA * self.mrt) + self.pickup_time

    def src_point(self):
        return (self.src_x, self.src_y)

    def dst_point(self):
        return (self.dst_x, self.dst_y)

    def points(self):
        return [self.src_point(), self.dst_point()]

    def __str__(self):
        src = f"SRC=({self.src_x}, {self.src_y})"
        dst = f"DST=({self.dst_x}, {self.dst_y})"
        drt = f"DRT={self.drt}"
        mrt = f"MRT={self.mrt}"
        picktime = f"PICKTIME={self.pickup_time}"
        sd = f"SERVICE_DURATION={self.service_duration} min"
        aos = f"AREA_OF_SERVICE={self.area_of_service} miles"

        return " | ".join([src, dst, drt, mrt, picktime, sd, aos])

def sqKm_to_sqMiles(km: float) -> float:
    return km * 0.386102

def getRequests(noof_customers, service_time_hrs, area_of_service_kms):
    service_time = service_time_hrs * 60 # minutes
    area_of_service = sqKm_to_sqMiles(area_of_service_kms) # square km
    filepath = f"DARPDATASET/Temportal-DS/nCustomers_{noof_customers}/Temporal_SD{service_time_hrs}hrs_SA{area_of_service_kms}km.txt"
    requests = []
    with open(filepath, 'r') as f:
        # first line is the depot
        for id, line in enumerate(f.readlines()):
            requests.append(Request(id, line, service_time, int(area_of_service)))
    return requests
