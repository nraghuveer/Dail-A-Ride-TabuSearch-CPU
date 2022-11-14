"""
Our goal is to construct cluster of requests which are close in respect to "adt" => average departure time
So we build a graph where each request is node and each vehicle is also a node
Arc weights is adt' is defined by (20) in the paper
(adt') is refered as `D bar ij`
"""
from scipy.optimize import linear_sum_assignment
import numpy as np

def build_graph(gts, noof_vehicles=2):
    r = len(gts.requests) # 0....r, r+1....r+n+1
    n = len(gts.requests) + noof_vehicles
    # r = 11, n = 10 + 3 + 1
    # 1...10,11,12,13
    def isV(x): return x >= r
    def isR(x): return not isV(x)

    graph = [[float('inf') for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # if both i and j are requests
            # use average departure time
            if isR(i) and isR(j):
                graph[i][j] = gts.adt(i, j)
            elif isV(i) and isV(j): # both are vehicles
                continue # inf will be value
            elif isV(i) and isR(j):
                # V * R
                t_0j = gts.travel_time(0, j)
                wait_time = max(gts.tw[j][0] - 0 - t_0j, 0)
                graph[i][j] = 0 + wait_time + t_0j + gts.d[j]
            elif isR(i) and isV(j):
                # R * V
                # end of time window at depot? is total ride time?
                t = gts.travel_time(-i, 0)
                graph[i][j] = gts.service_duration - gts.tw[-i][0] - t - gts.d[-i]
    return graph

def run_assignment_problem(gts):
    graph = build_graph(gts)
    # print(graph)
    graph = np.array(graph)
    rows, cols = linear_sum_assignment(graph, maximize=False)
    print(graph[rows, cols])
