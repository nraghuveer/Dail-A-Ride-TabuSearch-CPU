"""
3.3

For each cluster which includes a vehicle, apply a simple step-by-step procedure to produce a feasible route
Initially, the route consists of only depot.
Then for each request of cluster, the procedure attemps to insert the request
into the route, if failsm the request is set as unserved.


Paper1 - 3.4
Make feasible and infeasible solution

The solution obtained with assignment problem consistes of a set of nv
"""

# window tighting
# assign high values to all arcs which cannot be part of feasible solution
# . start depot to delivery nodes and vice-versa
# . arcs connecting nodes which are incompatible regarding their time-windows and time travel


