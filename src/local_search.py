# we are interested in low reduced costs

T_Gran = float('-inf')

# TODO: apply graph pruning and time window tightening
# tw at pickup => e = max(0, e_i-, T_trip - d_i+)

"""
Each iterator of the granula tabu search opimizaiton process reqires evaluation
of the whole granular neighborhood
This is similar to the original paper "making feasible solution"
"""


# maximum amount of time by which departure time from a node i can be delayed
# without violating time window and passenger ride time
def forward_slack_time(i, q, gts):
    # min of
    # (sum of
    #   (w + min of (end of tw - begging of service at j,
    #                   total trip - ride time of passenger whose destination is j)) for all j between i and q)
    Fi = float('inf')
    for j in range(i, q+1):
        for p in range(i+1, j+1):
            Fi = min(Fi, 100)
    return Fi

def make_feasible(n, m, routes, gts):
    pass


