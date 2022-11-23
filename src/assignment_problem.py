"""
Our goal is to construct cluster of requests which are close in respect to "adt" => average departure time
So we build a graph where each request is node and each vehicle is also a node
Arc weights is adt' is defined by (20) in the paper
(adt') is refered as `D bar ij`
"""
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from pyvis.network import Network
import numpy as np

def visualize_graph(n, m, rows, cols):
    net = Network()
    for x in range(n):
        net.add_node(str(x), label=str(x), group=1)
    for x in range(n, n+m):
        net.add_node(str(x), label=str(x), size=20, group=2)
    for u, v in zip(rows, cols):
        net.add_edge(str(u), str(v))
    net.show("net.html")

def build_initial_routes(n, m, rows, cols):
    routes = defaultdict(list)
    graph = defaultdict(list)
    for u, v in zip(rows, cols):
        graph[u].append(v)
        graph[v].append(u)

    def dfs(u, route, visited):
        for v in graph[u]:
            if v not in visited:
                visited.add(v)
                route.append(v)
                dfs(v, route, visited)

    # TODO: some routes might not be include any vehicle
    for v in range(n, n + m):
        route = routes[v]
        visited = set([v])
        dfs(v, route, visited)

    return routes


def build_graph(gts, n, m):
    N = n + m
    # r = 11, n = 10 + 3 + 1
    # 1...10,11,12,13
    def isV(x):
        return x >= n  # returns true if the node is vehicle

    def isR(x):
        return not isV(x)  # returns true if the node is request

    graph = [[float("inf") for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # if both i and j are requests
            # use average departure time
            elif isR(i) and isR(j):
                graph[i][j] = gts.adt(i, j)
            elif isV(i) and isV(j):  # both are vehicles
                continue  # inf will be value
            elif isV(i) and isR(j):
                # V * R
                t_0j = gts.travel_time(0, j)
                wait_time = max(gts.e(j) - gts.e(0) - t_0j, 0)
                graph[i][j] = gts.e(0) + wait_time + t_0j + gts.d[j]
            elif isR(i) and isV(j):
                # R * V
                # end of time window at depot? is total ride time?
                t = gts.travel_time(-i, 0)
                graph[i][j] = gts.service_duration - gts.e(-i) - t - gts.d[-i]
    return graph

def run_assignment_problem(gts):
    graph = build_graph(gts, len(gts.requests), 2)
    # print(graph)
    graph = np.array(graph)
    rows, cols = linear_sum_assignment(graph, maximize=False)
    # pprint(list(zip(rows, cols)))
    # set the arcs used in solution to 0
    for u, v in zip(rows, cols):
        graph[u, v] = 0

    visualize_graph(len(gts.requests), 2, rows, cols)
    routes = build_initial_routes(len(gts.requests), 2, rows, cols)
    print(routes)
    # for row in graph:
    #     print(row)
    # print(sum(map(len, routes)))
    return routes

