import argparse
from time import time
from typing import List, Dict, Tuple
from itertools import product
from parse_requests import Request, getRequests
from assignment_problem import run_assignment_problem, visualize_graph

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
        
    def __init__(self, noof_customers: int, service_duration: int, area_of_service: int,
                 noof_vehicles: int) -> None:
        self.service_duration = service_duration
        self.noof_customers = noof_customers
        self.area_of_service = area_of_service
        self.noof_vehciles = noof_vehicles
        self.V = [f"v{v}" for v in range(self.noof_vehciles)]
        self.T_route = self.service_duration
        # max ride of each passenger
        # IDEA: maybe we can do like alpha times * time from pickup to dropoff
        self.T_ride = int(0.75 * self.service_duration)
        self.requests: List[Request] = getRequests(self.noof_customers, self.service_duration, self.area_of_service)
        # there are total of n requests, so n nodes
        # 1....n
        # 0 and 2n+1 are the depot
        self.n = len(self.requests)
        self.m = self.noof_vehciles
        self.start_depot = 0
        self.end_depot = 2*self.n + 1
        self.Q = 4  # capacity of each vechicle

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
        for req in self.requests:
            self.w[req.id] = 0
            self.w[-req.id] = 150

    def e(self, x):
        return self.tw[x][0]

    def l(self, x):
        return self.tw[x][1]

    def start(self):
        start = time()
        routes = run_assignment_problem(self)
        print("*"*30)
        self.print_config()
        print(len(routes))
        for r in routes:
            if r and self.isV(r[0]):
                print(f"Main path => {r}")
            else:
                print(f"Sub tour => {r}")

        visualize_graph(self, routes)
        print(f"Total time = {time() - start} seconds")

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
        return abs(pone[0]-ptwo[0]) + abs(pone[1]-ptwo[1])

    def check_time_feasibility(self, seq: List[int]):
        # 7-10 => ensure correct arrival, service and departure time
        # 7 -> for i, j => arrival at j <= start of service at j
        # departure from i + time for i-j = Arrival time at j <= start of service at j
        A = {}
        B = {}
        D = {}
        i = seq[0]
        A[i] = self.e(i)
        B[i] = A[i] + self.w[i] # w[i] will be zero if the i is pickup
        D[i] = B[i] + self.d[i]
        for i, j in zip(seq, seq[1:]):
            # there might be some wait time if the node is pickup
            A[j] = D[i] + self.travel_time(i, j)
            B[j] = A[j] + self.w[j]
            D[j] = B[j] + self.d[j]

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

        #Equation 11
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

    # def get_service_quality(self, i_arr, i_dep):
    #     # end of time window at destination
    #     a = self.time_windows[i_dep]
    #     # start of service time at arrival
    #     e, _ = self.service_time[i_arr]
    #     t = self.trave_time[(i_arr, i_dep)]
    #     return (a - e) / t


    # def objective_function(self):
    #     cust = []
    #     N = self.arrivals[:]
    #     N.extend(self.departures[:])
    #     for i in self.arrivals:
    #        for v in self.vehicles:
    #            for j in N:
    #                 val = self.get_x(i, j, v) - self.get_service_quality(i, j)
    #                 cust.append(self.alpha_coefficient * val)
    #     return sum(cust)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--noof_customers', required=True, type=int, help='noof customers')
    parser.add_argument('-d', '--service_duration', type=int, help='service duration', required=True)
    parser.add_argument('-a', '--area_of_service', type=int, help='area of service', required=True)
    args = parser.parse_args()
    gts = GTS(args.noof_customers, args.service_duration, args.area_of_service, 2)
    gts.start()








