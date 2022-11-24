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

def add_fixed_node(net, n, x, gts, group):
    if x < n:
        p = gts.requests[x].src_point()
        label = f"{p} | " + str(gts.requests[x].pickup_time)
    else:
        p = gts.requests[0].src_point()
        label = f"{p} | " + str(gts.requests[0].pickup_time)
    xp, yp = p
    net.add_node(str(x), label=label, size=20, x=xp*200,
                 y=yp*200, physics=False, group=group)

def visualize_graph(gts, routes):
    n = gts.n
    net = Network()
    added = set()
    for g, route in enumerate(routes, 1):
        for u, v in zip(route, route[1:]):
            if u not in added:
                add_fixed_node(net, n, u, gts, group=g)
                added.add(u)
            if v not in added:
                add_fixed_node(net, n, v, gts, group=g)
                added.add(v)
            net.add_edge(str(u), str(v))
    net.show("net.html")

def build_paths(n, m, rows, cols):
    graph = defaultdict(list)
    for u, v in zip(rows, cols):
        graph[u].append(v)

    def dfs(u, visited, path):
        path.append(u)
        for v in graph[u]:
            if v not in visited:
                visited.add(v)
                dfs(v, visited, path)
        return
        
    paths = []
    visited = set()
    for v in range(n, n+m):
        if v in visited:
            continue
        path = []
        dfs(v, visited, path)
        paths.append(path)

    for u in range(n):
        if u in visited:
            continue
        path = []
        dfs(u, visited, path)
        paths.append(path)
    return paths
   
def build_graph(gts):
    n, m = gts.n, gts.m
    N = n + m
    # r = 11, n = 10 + 3 + 1
    # 1...10,11,12,13
    graph = [[float("inf") for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # if both i and j are requests
            # use average departure time
            elif gts.isR(i) and gts.isR(j):
                graph[i][j] = gts.adt(i, j)
            elif gts.isV(i) and gts.isV(j):  # both are vehicles
                continue  # inf will be value
            elif gts.isV(i) and gts.isR(j):
                # V * R
                t_0j = gts.travel_time(0, j)
                wait_time = max(gts.e(j) - gts.e(0) - t_0j, 0)
                graph[i][j] = gts.e(0) + wait_time + t_0j + gts.d[j]
            elif gts.isR(i) and gts.isV(j):
                # R * V
                # end of time window at depot? is total ride time?
                t = gts.travel_time(-i, 0)
                graph[i][j] = gts.service_duration - gts.e(-i) - t - gts.d[-i]
    return graph

def run_assignment_problem(gts):
    n, m = gts.n, gts.m
    graph = build_graph(gts)
    graph = np.array(graph)
    rows, cols = linear_sum_assignment(graph, maximize=False)
    # set the arcs used in solution to 0
    for u, v in zip(rows, cols):
        graph[u, v] = 0

    routes = build_paths(n, m, rows, cols)
    return routes

