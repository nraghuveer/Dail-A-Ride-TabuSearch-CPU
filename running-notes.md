* A main path is a chain of nodes (customers) containing the node depot
* The subtours involve only customers, thus are easily inserted into the main path
* Chains of customers reperesent a first step towards a feasible solution for the original problem, since all they are missing is delivery node
* There are two ways to insert delivery nodes.


## Constraints

C2-4 => each request is served by at most one vehicle
C5-6 => feasibility of load

The assignment problem provides the reduced cost value for each arc (i, j) calculated
as `c_ij = D_cap_ij - u_i - v_j`, where u_i and v_j are dual variabels of constraints (22), (23)

Graph
    left is i, right is j
    if i, j are requests -> average departure time
    if i, j are vehicles -> inf
    if i is vehicle and j is request -> e0 + t0j+ + wj+ + dj+
        assigning a vehicle to request
        e0 => start time widow at depot
        t_0j+ => time to travel from depot to j+ (pickup)
        wj+ => wait time at pickup of j
        dj+ => service time at pickiup of j
    if i is request and j is vehicle ->
        assigning a request to vehicle

n + m nodes => so (n+m)^2 
