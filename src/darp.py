import argparse
import csv
from typing import List, Dict, Tuple
from parse_requests import Request, getRequests
from gpu import GPU

Point = Tuple[int, int]


class DARP:
    def print_config(self):
        print(f"number of requests => {self.n}")
        print(f"number of vehicles => {self.m}")
        print(f"Capacity of each vechicle => {self.Q}")
        print(f"area of service => {self.area_of_service} sq-kms")
        print(f"service duration => {self.service_duration} hrs")

    def get_config(self):
        return {
            "requests": self.n,
            "vehicles": self.m,
            "capacity": self.Q,
            "areaOfService": self.area_of_service,
            "serviceDuration": self.service_duration
        }

    def __init__(self, noof_customers: int, service_duration: int, area_of_service: int,
                 noof_vehicles: int, vechicleCapacity: int) -> None:
        self.service_duration = service_duration
        self.noof_customers = noof_customers
        self.area_of_service = area_of_service
        self.noof_vehciles = noof_vehicles
        self.V = [f"v{v}" for v in range(self.noof_vehciles)]
        self.T_route = int(self.service_duration * 60 * 60 * 0.50)  # convert hrs to seconds
        # max ride of each passenger
        # IDEA: maybe we can do like alpha times * time from pickup to dropoff
        self.T_ride = 0.40 * self.T_route
        self.requests: List[Request] = getRequests(self.noof_customers, self.service_duration, self.area_of_service)
        # there are total of n requests, so n nodes
        # 1....n
        # 0 and 2n+1 are the depot
        self.n = len(self.requests)
        self.m = self.noof_vehciles
        self.start_depot = 0
        self.end_depot = 2 * self.n + 1
        self.Q = vechicleCapacity  # capacity of each vechicle

        # build coordinates
        # each request has both pickup and dropoff coordinates
        # use +req.id as pickup as -req.id as dropoff
        self.coords: Dict[int, Point] = {}
        for req in self.requests[1:]:
            self.coords[req.id] = req.src_point()
            self.coords[-req.id] = req.dst_point()
        self.coords[self.start_depot] = self.requests[0].src_point()
        self.coords[self.end_depot] = self.requests[0].src_point()

        # service time at each node -> in minutes
        # this is something we came up with
        self.d: Dict[int, int] = {}
        for req in self.requests[1:]:
            self.d[req.id] = 2
            self.d[-req.id] = 2
        self.d[self.start_depot] = 2
        self.d[self.end_depot] = 2

        # load to carry
        self.q: Dict[int, int] = {}
        for req in self.requests[1:]:
            self.q[req.id] = req.load
            self.q[-req.id] = -req.load
        self.q[self.start_depot] = 0
        self.q[self.end_depot] = 0

        """
        [e, l] represents a time window
        DAR system should pickup/dropoff customer
        at in this time window at respective node
        """
        tw = {}
        # add 5 min time window
        offset = int(5 * 60)
        for req in self.requests[1:]:
            id = req.id
            tw[id] = (req.pickup_time, req.pickup_time + offset)
            tw[-id] = (req.dropoff_time, req.dropoff_time + offset)
        tw[self.start_depot] = (0, offset)
        tw[self.end_depot] = (self.T_route, self.T_route + offset)
        self.tw = tw

        """ wait times -> for now keep it 150 seconds, there is no wait times for drop nodes"""
        self.w = {}
        self.w[self.start_depot] = self.w[self.end_depot] = 0
        for req in self.requests:
            self.w[req.id] = 0
            self.w[-req.id] = 150

    def e(self, x):
        return self.tw[x][0]

    def l(self, x):
        return self.tw[x][1]

    def travel_time(self, one: int, two: int):
        pone = self.coords[one]
        ptwo = self.coords[two]
        return abs(pone[0] - ptwo[0]) + abs(pone[1] - ptwo[1])

    def check_time_feasibility(self, seq: List[int]):
        # 7-10 => ensure correct arrival, service and departure time
        # 7 -> for i, j => arrival at j <= start of service at j
        # departure from i + time for i-j = Arrival time at j <= start of service at j
        A = {}
        B = {}
        D = {}
        i = seq[0]
        A[i] = self.e(i)
        B[i] = A[i] + self.w[i]  # w[i] will be zero if the i is pickup
        D[i] = B[i] + self.d[i]
        for i, j in zip(seq, seq[1:]):
            # there might be some wait time if the node is pickup
            A[j] = D[i] + self.travel_time(i, j)
            B[j] = A[j] + self.w[j]
            D[j] = B[j] + self.d[j]

        # print(seq)
        # pprint(A)
        # pprint(B)
        # pprint(D)
        # print("\n\n")
        #
        # Equation 7
        for j in seq[1:]:
            if not A[j] <= B[j]:
                return False

        # Equation 8 => only for pickup nodes
        # time taken for start to j should be less than arrival of j and begging of j
        for i in seq:
            if i < 0:
                continue
            # assume we departure from D at 0th
            t = 0 + self.travel_time(self.start_depot, i)
            if not (t <= A[i] <= B[i]):
                return False

        # Equation 9 => only for dropoff
        for i in seq:
            if i >= 0:
                continue
            if not self.travel_time(i, self.end_depot) <= self.l(self.end_depot):
                return False

        # Equation 10
        for i in seq:
            if not (self.e(i) <= B[i] <= self.l(i)):
                return False

        # Equation 11
        for i in seq:
            if i >= 0:
                continue
            # begging of service at droff - departure from pickup should be <= max ride time
            if not B[i] - D[-i] <= self.T_ride:
                return False

        # TODO: Equation 12 => seems like we cannot apply this here
        return True

    def check_load_feasiblity(self, seq: List[int]):
        # CHECKS constraints 5, 6
        # load wheh leaving node i
        y: Dict[int, int] = {}
        y[seq[0]] = self.q[seq[0]]
        for prev, cur in zip(seq, seq[1:]):
            y[cur] = y[prev] + self.q[cur]

        # EQUATION 5, 6
        for i, j in zip(seq, seq[1:]):
            if not (y[i] + self.q[j] <= y[j]):
                return False
            if not (self.q[i] <= y[i] <= self.Q):
                return False
        return True

    def isV(self, x):
        return x >= self.n  # returns true if the node is vehicle

    def isR(self, x):
        return not self.isV(x)  # returns true if the node is request

    def writeToBenchmarkFile(self, filename, benchmark):
        print(f"Writing benchmarks to ", filename)
        with open(filename, 'a') as f:
            w = csv.DictWriter(f, fieldnames=list(benchmark.keys()))
            w.writerow(benchmark)

    def objective_function(self, route, A, B, D, y):
        # EQ:1 in the paper
        w1 = 8
        w2 = 3
        w3 = 1
        w4 = 1
        w5 = 1
        alpha = 10000

        def c(): # routing costs
            total = 0
            for prev, cur in zip(route, route[1:]):
                total += self.travel_time(prev, cur)
            return total

        def r(): # excess ride time
            pickups = [x for x in route if x > 0]
            return sum(B[-i] - D[i] - self.travel_time(i, -i) for i in pickups)

        def l():  # wait time of passengers on board
            nodes = [x for x in route if x not in [self.start_depot, self.end_depot]]
            return sum(self.w[i]*(y[i] - self.q[i]) for i in nodes)

        def g(): # route durations
            return

        def e():
            return 0

        def k():
            return 0

        return w1 * c() + w2 * r() + w3 * l() + w4 * g() + w5 * e() + alpha * k()

    def evaluate_route(self, route: List[int]) -> bool:
        # Eight-step evaluation
        # compute following for each node
        """
        Departure-prev ...Ride...Arrival..service..wait...Departure
        """
        def calc(D0):
            A: Dict[int, int] = {route[0]: D0}  # Arrival time
            w: Dict[int, int] = {route[0]: 0}  # wait times
            B: Dict[int, int] = {route[0]: D0}  # Beginning of service
            D: Dict[int, int] = {route[0]: D0}  # Departure times
            y: Dict[int, int] = {route[0]: 0}  # load at the time of leaving node x
            for prev, i in zip(route, route[1:]):
                # Arrival at i = Departure from previous node + travel time
                A[i] = D[prev] + self.travel_time(prev, i)
                # Beging service after the wait-time....
                # TODO: why do we have to wait here?
                B[i] = A[i]  # + self.w[i]
                # after the service -> departure
                D[i] = B[i] + self.d[i]
                w[i] = B[i] - A[i]
                # load after leave i, since self.q can have negative values for the
                # dropoff nodes, this statement covers all cases
                y[i] = y[prev] + self.q[i]
            return A, w, B, D, y

        A, w, B, D, y = calc(self.e(route[0]))
        for i in route:
            if not B[i] <= self.l(i) and y[i] <= self.Q:
                return False

        # make the moves
        for j_idx, j in enumerate(route):
            Fj = self.forward_slack_time(route, j, w, B, D[route[-1]])
            w[j] = w[j] + min(Fj, sum(w[p] for p in route[j_idx+1:-1]))
            B[j] = A[j] + w[j]
            D[j] = B[j] + self.d[j]
            for prev, i in zip(route[j_idx:], route[j_idx+1: ]):
                # TODO: duplicate code
                A[i] = D[prev] + self.travel_time(prev, i)
                # Begin service after the wait-time....
                B[i] = A[i]  # + self.w[i]
                # after the service -> departure
                D[i] = B[i] + self.d[i]
                w[i] = B[i] - A[i]
            Ti_ride = [A[-abs(i)] - B[abs(i)] for i in route[j_idx+1:-1]]
            if all([ride <= self.T_ride for ride in Ti_ride]):
                return True
        return False

    def forward_slack_time(self, route: List[int], i: int, w: Dict[int, int], B: Dict[int, int], T_Trip: int):
        # so this is basically cummulative waiting times and the difference between end of time window and start of
        # it is min of all slacks at node j
        # reference: https://logistik.bwl.uni-mainz.de/files/2018/12/LM-2015-01-revised.pdf
        q = route[-1]
        res = float('inf')
        i_idx = route.index(i)
        for j_idx in range(i_idx, len(route)):
            # j is some point between and i and q
            # calculate all the waittimes between i and j
            wait_times = [w[route[x]] for x in range(i_idx, j_idx + 1)]
            tw_slack = self.l(route[j_idx]) - B[route[j_idx]]
            res = min(res, sum(wait_times) + tw_slack)
        return res

    def start(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--noof_customers', required=True, type=int, help='noof customers')
    parser.add_argument('-d', '--service_duration', type=int, help='service duration', required=True)
    parser.add_argument('-a', '--area_of_service', type=int, help='area of service', required=True)
    parser.add_argument('-v', '--noof_vehicles', required=True, type=int, help='noof vehicles')
    parser.add_argument('-Q', '--vehicle_capacity', required=True, type=int, help='vehicle capacity')
    args = parser.parse_args()
    darp = DARP(args.noof_customers, args.service_duration, args.area_of_service, args.noof_vehicles,
              args.vehicle_capacity)
    benchmarking = darp.get_config()
    gpu = GPU(darp, local_search_iterations=200, local_search_size=700)
    init_solution = gpu.construction_kernel()
    print(init_solution)
    solution = gpu.local_search_kernel(init_solution)
    print(solution)
    print('done')

