from pyscipopt.scip import Model, is_memory_freed, quicksum

def is_optimized_mode():
    s = Model()
    return is_memory_freed()

def build_random_model_1(disable_heuristics=True, disable_separators=True, disable_presolve=False, node_lim=2000):
    scip = Model()
    if disable_heuristics:
        scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    if disable_separators:
        scip.setSeparating(SCIP_PARAMSETTING.OFF)
    if disable_presolve:
        scip.setPresolve(SCIP_PARAMSETTING.OFF)
    scip.setLongintParam("limits/nodes", node_lim)

    x0 = scip.addVar(lb=-2, ub=4)
    r1 = scip.addVar()
    r2 = scip.addVar()
    y0 = scip.addVar(lb=3)
    t = scip.addVar(lb=None)
    l = scip.addVar(vtype="I", lb=-9, ub=18)
    u = scip.addVar(vtype="I", lb=-3, ub=99)

    more_vars = []
    for i in range(100):
        more_vars.append(scip.addVar(vtype="I", lb=-12, ub=40))
        scip.addCons(quicksum(v for v in more_vars) <= (40 - i) * quicksum(v for v in more_vars[::2]))

    for i in range(100):
        more_vars.append(scip.addVar(vtype="I", lb=-52, ub=10))
        scip.addCons(quicksum(v for v in more_vars[50::2]) <= (40 - i) * quicksum(v for v in more_vars[405::2]))

    scip.addCons(r1 >= x0)
    scip.addCons(r2 >= -x0)
    scip.addCons(y0 == r1 + r2)
    scip.addCons(t + l + 7 * u <= 300)
    scip.addCons(t >= quicksum(v for v in more_vars[::3]) - 10 * more_vars[5] + 5 * more_vars[9])
    scip.addCons(more_vars[3] >= l + 2)
    scip.addCons(7 <= quicksum(v for v in more_vars[::4]) - x0)
    scip.addCons(quicksum(v for v in more_vars[::2]) + l <= quicksum(v for v in more_vars[::4]))

    scip.setObjective(t - quicksum(j * v for j, v in enumerate(more_vars[20:-40])))

    return scip

def build_minimum_hypergraph_model(disable_heuristics=Fasle, disable_separators=False, disable_presolve=False):

    scip = Model()
    if disable_heuristics:
        scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    if disable_separators:
        scip.setSeparating(SCIP_PARAMSETTING.OFF)
    if disable_presolve:
        scip.setPresolve(SCIP_PARAMSETTING.OFF)

    # Make a basic minimum spanning hypertree problem
    # Let's construct a problem with 15 vertices and 40 hyperedges. The hyperedges are our variables.
    v = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    e = {}
    for i in range(40):
        e[i] = scip.addVar(vtype='B', name='hyperedge_{}'.format(i))

    # Construct a dummy incident matrix
    A = [[1, 2, 3], [2, 3, 4, 5], [4, 9], [7, 8, 9], [0, 8, 9],
         [1, 6, 8], [0, 1, 2, 9], [0, 3, 5, 7, 8], [2, 3], [6, 9],
         [5, 8], [1, 9], [2, 7, 8, 9], [3, 8], [2, 4],
         [0, 1], [0, 1, 4], [2, 5], [1, 6, 7, 8], [1, 3, 4, 7, 9],
         [11, 14], [0, 2, 14], [2, 7, 8, 10], [0, 7, 10, 14], [1, 6, 11],
         [5, 8, 12], [3, 4, 14], [0, 12], [4, 8, 12], [4, 7, 9, 11, 14],
         [3, 12, 13], [2, 3, 4, 7, 11, 14], [0, 5, 10], [2, 7, 13], [4, 9, 14],
         [7, 8, 10], [10, 13], [3, 6, 11], [2, 8, 9, 11], [3, 13]]

    # Create a cost vector for each hyperedge
    c = [2.5, 2.9, 3.2, 7, 1.2, 0.5,
         8.6, 9, 6.7, 0.3, 4,
         0.9, 1.8, 6.7, 3, 2.1,
         1.8, 1.9, 0.5, 4.3, 5.6,
         3.8, 4.6, 4.1, 1.8, 2.5,
         3.2, 3.1, 0.5, 1.8, 9.2,
         2.5, 6.4, 2.1, 1.9, 2.7,
         1.6, 0.7, 8.2, 7.9, 3]

    # Add constraint that your hypertree touches all vertices
    scip.addCons(quicksum((len(A[i]) - 1) * e[i] for i in range(len(A))) == len(v) - 1)

    # Now add the sub-tour elimination constraints.
    for i in range(2, len(v) + 1):
        for combination in itertools.combinations(v, i):
            scip.addCons(quicksum(max(len(set(combination) & set(A[j])) - 1, 0) * e[j] for j in range(len(A))) <= i - 1,
                         name='cons_{}'.format(combination))

    # Add objective to minimise the cost
    scip.setObjective(quicksum(c[i] * e[i] for i in range(len(A))), sense='minimize')

    return scip
