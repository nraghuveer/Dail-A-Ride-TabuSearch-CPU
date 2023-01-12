import os
from functools import reduce
import argparse
import csv
from pprint import pprint
from time import time
from typing import List, Dict, Tuple
from itertools import product
from parse_requests import Request, getRequests
from assignment_problem import run_assignment_problem, visualize_graph, visualize_3d

Point = Tuple[int, int]

"""
earliest departure -> whats the earliest time that the customer is avaiable for pick or we can pikup customer
    we can after this time (pickup)
latest departure -> whats the atmost time we cna dropoff
    we can do before this time (dropoff)
"""

"""
Granular tabu search implemenation of dail-a-ride algorithm
GTS is a TS with a particular focus on the local search phase
The size of GTS is regulated by a granular threshold

It is based on the idea that the solutions rarely contain long edges
the algorithm explores only edges with a length up to a certain threshold
during the local seaerch phase.

Simple Moves:
1. Move a request from a route into another one
2. insert an unserved request in a route, or remove a request from route
A granular neighborhood, the only moves allowed are those with reduced cost
c_ij < T_Gran
"""


# STATIC DAR with time windows and a fleet of fixed size
# Each request corresponds to a single passenger and
# object function maximized firstly the number of customers served,
# then the level of service provided on average


class GTS:
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

    def start(self):
        clusters = self.generate_clusters()
        routes, unserved = self.generate_routes_from_clusters(clusters)
        print(routes)
        print(unserved)

    def generate_clusters(self):
        benchmarking = self.get_config()
        start = time()
        routes = run_assignment_problem(self)
        benchmarking["assignmentProblemTime"] = time() - start
        print("*" * 30)
        self.print_config()
        print("*" * 30)

        def print_node(x):
            if self.isV(x):
                return x
            p = self.requests[x]
            px, py = p.src_point()
            return f"(({px}, {py}), {self.e(x)} - {self.l(x)})"

        for i, r in enumerate(routes):
            if r and self.isV(r[0]):
                # print(f"{i} - Main path => {list(map(print_node, r))}")
                print(f"{i} - MainPath => {r}")
            else:
                # print(f"{i} Sub tour => {list(map(print_node, r))}")
                print(f"{i} - SubTour => {r}")
        print("*" * 30)
        benchmarking['totalTime'] = time() - start
        print(f"Total time = {benchmarking['totalTime']} seconds")
        filename = os.environ.get("DARP_BENCHMARKFILE", "benchmark.csv")
        self.writeToBenchmarkFile(filename, benchmarking)
        # visualize_3d(gts, routes)
        visualize_graph(self, routes)
        return routes

    def print_adt(self):
        # calculate _D_ij for all node combinations
        results = []
        for pointI, pointJ in product(self.requests, self.requests[1:]):
            if pointI.id == pointJ.id:
                continue
            res = self.adt(pointI.id, pointJ.id)
            results.append((f"{pointI.points()}", f"{pointJ.points()}", res))

        results = sorted(results, key=lambda x: x[2])
        print("Average departure time, how close they are in spatial and temporal")
        print("\n".join([f"(from={x[0]}, to={x[1]}, d={x[2]})" for x in results]))

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

    def adt(self, i: int, j: int):
        """ returns D~_ij """
        """ This is used measure spatial and temoral distance between two requests i and j"""

        # Following are possible sequences
        # i+, i-, j+, j- => pi1
        # i+, j+, i-, j- => pi2
        # i+, j+, j-, i- => pi3
        def for_sequence(one, two, three, four):
            if one < 0:
                raise ValueError("the start point in the sequence should be always be arrival node")

            return self.e(one) + self.d[one] + \
                self.travel_time(one, two) + self.d[two] + \
                self.travel_time(two, three) + self.d[three] + \
                self.travel_time(three, four) + self.d[four]

        # since j+ is arrival node here, if we arrive early than the actual arrival
        # we have to wait till the arrival
        # TODO: D_pi hosuld be set to inf, if the time or load constraints are not respected
        # for constraints, consider the path to be the sequence and a vehicle is used
        # and then check for constraints
        wait_time_at_j = self.w[j]
        seq1 = [i, -i, j, -j]
        seq2 = [i, j, -i, -j]
        seq3 = [i, j, -j, -i]

        def check_constraints(seq: List[int]):
            return self.check_load_feasiblity(seq) and self.check_time_feasibility(seq)

        if check_constraints(seq1):
            D_pi1 = for_sequence(*seq1) + wait_time_at_j
        else:
            D_pi1 = float("inf")

        if check_constraints(seq2):
            D_pi2 = for_sequence(*seq2) + wait_time_at_j
        else:
            D_pi2 = float("inf")

        if check_constraints(seq3):
            D_pi3 = for_sequence(*seq3) + wait_time_at_j
        else:
            D_pi3 = float("inf")

        return sum(D for D in [D_pi1, D_pi2, D_pi3] if D != float('inf'))

    def writeToBenchmarkFile(self, filename, benchmark):
        print(f"Writing benchmarks to ", filename)
        with open(filename, 'a') as f:
            w = csv.DictWriter(f, fieldnames=list(benchmark.keys()))
            w.writerow(benchmark)

    # def get_service_quality(self, i, j):
    #     # end of time window at destination
    #     a = self.e(i)
    #     e = self.l(j)
    #     # start of service time at arrival
    #     t = self.travel_time(i, -i)
    #     return (a - e) / t

    # def objective_function(self):
    #     cust = []
    #     N = self.arrivals[:]
    #     N.extend(self.departures[:])
    #     for i in self.arrivals:
    #        for v in self.vehicles:
    #            for j in N:
    #                 val = self.get_x(i, j, v) - self.get_service_quality(i)
    #                 cust.append(self.alpha_coefficient * val)
    #     return sum(cust)

    # will be applied on whole solution
    # not just on a route
    def objective_function(self, B, D):
        # EQ:1 in the paper
        w1 = 8
        w2 = 3
        w3 = 1
        w4 = 1
        w5 = 1
        alpha = 10000

        def c():
            return sum(self.travel_time(i, j) for j in self.requests for i in self.requests if i != j)

        def r():
            return 0

        def l():
            return 0

        def g():
            return 0

        def e():
            return 0

        def k():
            return 0

        return w1 * c() + w2 * r() + w3 * l() + w4 * g() + w5 * e() + alpha * k()

    def possible_routes_with_changing_dropoffs(_, route, req) -> List[int]:
        """Given a route, and a dropoff, yields all the possible routes where dropoff is changed to all positions
        after its respective pickup"""
        route.remove(-req)
        pickup_idx = route.index(req)

        # consider all valid positions to insert this dropoff and change the route
        for idx in range(pickup_idx+1, len(route)-1):
            new_route = route[:]
            new_route.insert(idx, -req)
            yield new_route

    def generate_route_from_cluster(self, cluster):
        # remove start, end deport and vehicles from the route
        requests = [x for x in cluster if not self.isV(x) and x not in [self.start_depot, self.end_depot]]

        # start with just depot in the route
        # TODO: we are using the same order from cluster, should we change it?

        # the cluster is set of pickup points, generate a initial route
        # with start,x,-x, y, -y.......end
        def reduce_fn(res, cur):
            res.extend(cur)
            return res
        route = [self.start_depot] + reduce(reduce_fn, [(x, -x) for x in requests], []) + [self.end_depot]
        print(route)
        unserved = []
        # start from the first pickup node and check for local feasiblilty: tw of delivery node under consideration
        for req in requests[1:-1]:
            cheap_route = (float('inf'), None)
            for new_route in self.possible_routes_with_changing_dropoffs(route[:], req):
                if self.evaluate_route(new_route):
                    if cheap_route[0] < 0:
                        cheap_route = (0, new_route)
            if cheap_route[1] == None:
                # set as unserved
                route.remove(req)
                route.remove(-req)
                unserved.append(req)

        return route, unserved

    def generate_routes_from_clusters(self, clusters):
        routes = []
        unserved = []
        for cluster in clusters:
            if not self.isV(cluster[0]):
                unserved.extend(cluster)
                continue
            route, unserved_ = self.generate_route_from_cluster(cluster)
            if unserved_:
                unserved.extend(unserved_)
            routes.append(route)
        return routes, unserved

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

        # computer F0 => amount by which we can delay starting from the first node in the route
        # concept is basically use the time that we HAD to wait as DELAY
        # gather this pieces of time nodes from node after i in the route
        T_Trip = D[route[-1]]
        F0 = self.forward_slack_time(route, 0, w, B, T_Trip)
        # delay the departure from the start node
        D0 = self.e(route[0]) + min(F0, sum(w[p] for p in route[1:-1]))
        A, w, B, D, y = calc(D0)
        Ti_ride = {i: A[-abs(i)] - B[abs(i)] for i in route[1:-1]}
        if all(Ti_ride[i] <= self.T_ride for i in route[1:-1]):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--noof_customers', required=True, type=int, help='noof customers')
    parser.add_argument('-d', '--service_duration', type=int, help='service duration', required=True)
    parser.add_argument('-a', '--area_of_service', type=int, help='area of service', required=True)
    parser.add_argument('-v', '--noof_vehicles', required=True, type=int, help='noof vehicles')
    parser.add_argument('-Q', '--vehicle_capacity', required=True, type=int, help='vehicle capacity')
    args = parser.parse_args()
    gts = GTS(args.noof_customers, args.service_duration, args.area_of_service, args.noof_vehicles,
              args.vehicle_capacity)
    # gts.generate_clusters()
    gts.start()