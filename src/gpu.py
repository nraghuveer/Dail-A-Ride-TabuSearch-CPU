import concurrent.futures
from typing import List, Dict, Tuple

MAX_GENERATION = 100000

class ConstructionKernel:
    def __init__(self, gpu: 'GPU', seed, noof_requests, noof_vehicles):
        self.seed = seed
        self.gpu = gpu
        self.n = noof_requests
        self.m = noof_vehicles
        self.requests = list(range(1, self.n))
        self.vehicles = list(range(self.n, self.n+self.m+1))

    def generate(self):
        import random
        random.seed(self.seed)
        requests = random.sample(self.requests, self.n-1)
        routes = {m: [] for m in self.vehicles}
        for i in requests:
            k = random.choice(self.vehicles)
            # [1,2,3]
            route_len = len(routes[k])
            p1 = random.choice(range(route_len + 1))
            p2 = random.choice(range(p1+1, route_len+2))
            routes[k].insert(p1, i)
            routes[k].insert(p2, -i)
        optimization_value = optimization_fn(self.gpu, routes)
        # print(self.gpu.darp.n, str(self.gpu.darp.end_depot))
        print("############ "+ f"{self.seed} - {optimization_value}" +" ################")
        for m in routes:
            print(routes[m])
        return routes, optimization_value


def optimization_fn(gpu: 'GPU', routes: Dict[int, List[int]]):
    """ Given a route, returns the value of the optimization function"""

    # we need to calc arrival time, wait time, beggining for service and departure times for each route
    def calc_route_values(raw_route) -> Dict[str, Dict[int, int]]:
        # add depots to the route
        route = [0] + raw_route + [gpu.darp.end_depot]
        A: Dict[int, int] = {0: 0}  # Arrival time
        w: Dict[int, int] = {0: 0}  # wait times
        B: Dict[int, int] = {0: 0}  # Beginning of service
        D: Dict[int, int] = {0: 0}  # Departure times
        y: Dict[int, int] = {0: 0}  # load at the time of leaving node x

        for prev, i in zip(route, route[1:]):
            # Arrival at i = Departure from previous node + travel time
            A[i] = D[prev] + gpu.darp.travel_time(prev, i)
            # Beging service after the wait-time....
            # TODO: why do we have to wait here?
            B[i] = A[i]  # + self.w[i]
            # after the service -> departure
            D[i] = B[i] + gpu.darp.d[i]
            w[i] = B[i] - A[i]
            # load after leave i, since self.q can have negative values for the
            # dropoff nodes, this statement covers all cases
            y[i] = y[prev] + gpu.darp.q[i]
        return {
            'A': A, 'w': w, 'B': B, 'D': D, 'y': y
        }

    route_values = {k: calc_route_values(routes[k]) for k in routes}

    # c(x) + alpha * q(x) + beta * d(x) + gamma * w(x) + delta * t(x)

    # c(x) => total cost => total distance travelled by all vehicles
    def c_of_route(route):
        return sum(gpu.darp.travel_time(x, y) for x, y in zip(route, route[1:]))
    c = sum(c_of_route(route) for route in routes.values())

    # q(x) => load constraint voilation
    def max_capacity_in_route(k):
        return max(route_values[k]['y'].values())
    q = sum(max(max_capacity_in_route(k) - gpu.darp.Q, 0) for k in routes)

    # d(x) => happens when a vehicle k exceeds its duration limit
    def duration_of_route(k):
        return route_values[k]['D'][gpu.darp.n*2 + 1]
    d = sum(max(duration_of_route(k) - gpu.darp.T_route, 0) for k in routes)

    # w(x) => happends when time constraints to pickup and dropoff is voilated
    def requests_from_route(k):
        route = set([abs(i) for i in routes[k]])
        return route

    def late_quantity(k, i):
        """ returns how much late in pickup and dropoff """
        Bi = route_values[k]['B'][i]
        li = gpu.darp.l(i)
        Bi_minus = route_values[k]['B'][-i]
        li_minus = gpu.darp.l(-i)
        return max(Bi - li, 0) + max(Bi_minus - li_minus, 0)
    w = sum(late_quantity(k, i) for k in routes for i in requests_from_route(k))

    # t(x) => happens when a passenger is transported for a longer time than the ride time limit L

    # TODO: all the penality coefficients = 1
    return c + q + d + w


class GPU:
    def __init__(self, darp):
        self.darp = darp

    def construction_kernel(self):
        def map_fn(args):
            gpu, seed, n, m = args
            k = ConstructionKernel(gpu, seed, n, m)
            return k.generate()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            args = [(self, seed, self.darp.n, self.darp.m) for seed in range(MAX_GENERATION)]
            best_route = (float('inf'), None)
            for ret in executor.map(map_fn, args):
                print(ret)
                route, optimizationVal = ret
                if optimizationVal < best_route[0]:
                    best_route = (optimizationVal, route)

            print("best route")
            print(best_route[1])
            print(best_route[0])



