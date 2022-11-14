from typing import List, Dict, Tuple
from itertools import product
from parse_requests import Request, getRequests
from assignment_problem import run_assignment_problem

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


class GTS:
    def __init__(self) -> None:
        self.service_duration = 120 # for now
        self.requests: List[Request] = getRequests()
        # build coordinates
        # each request has both pickup and dropoff coordinates
        # use +req.id as pickup as -req.id as dropoff
        self.coords: Dict[int, Point] = {}
        for req in self.requests:
            self.coords[req.id] = req.src_point()
            self.coords[-req.id] = req.dst_point()

        # service time at each node -> in minutes
        # this is something we came up with
        self.d: Dict[str, int] = {}
        for req in self.requests:
            self.d[req.id] = 2
            self.d[-req.id] = 2

        """
        [e, l] represents a time window
        DAR system should pickup/dropoff customer
        at in this time window at respective node
        """
        tw = {}
        # add 5 min time window
        offset = int(5 * 60)
        for req in self.requests:
            id = req.id
            tw[id] = (req.pickup_time, req.pickup_time + offset)
            tw[-id] = (req.dropoff_time, req.dropoff_time + offset)
        self.tw = tw

        """ wait times -> for now keep it 150 seconds, there is no wait times for drop nodes"""
        self.w = {req.id: 150 for req in self.requests}

    def start(self):
        run_assignment_problem(self)

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

    def adt(self, i: int, j: int):
        """ returns D~_ij """
        """ This is used measure spatial and temoral distance between two requests i and j"""

        # Following are possible sequences
        # i+, i-, j+, j- => pi1
        # i+, j+, i-, j- => pi2
        # i+, j+, j-, i- => pi3
        def for_sequence(one, two, three, four):
            start_tw = self.tw[one][0]
            return start_tw + self.d[one] + \
                self.travel_time(one, two) + self.d[two] + \
                self.travel_time(two, three) + self.d[three] + \
                self.travel_time(three, four) + self.d[four]

        # since j+ is arrival node here, if we arrive early than the actual arrival
        # we have to wait till the arrival
        # TODO: fix this
        wait_time_at_j = 0
        D_pi1 = for_sequence(i, -i, j, -j) + wait_time_at_j
        D_pi2 = for_sequence(i, j, -i, -j) + wait_time_at_j
        D_pi3 = for_sequence(i, j, -j, -i) + wait_time_at_j

        # TODO: check if load or time constraints are respected at each node
        # for now, assume all sequences are feasible
        return D_pi1 + D_pi2 + D_pi3

    def get_service_quality(self, i_arr, i_dep):
        # end of time window at destination
        a = self.time_windows[i_dep]

        # start of service time at arrival
        e, _ = self.service_time[i_arr]
        t = self.trave_time[(i_arr, i_dep)]
        return (a - e) / t


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
    gts = GTS()
    gts.start()








