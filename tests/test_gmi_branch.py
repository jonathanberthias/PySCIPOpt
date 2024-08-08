from pyscipopt import Model, Branchrule, SCIP_RESULT, quicksum, SCIP_PARAMSETTING
import math


class GMIBranchingRule(Branchrule):

    def __init__(self, scip, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scip = scip

    def getGMISplitFromRow(self, cols, rows, binvrow, binvarow, primsol, lp_pos_col):

        # initialise
        splitcoefs = [0] * len(cols)

        # get scip
        scip = self.model

        # Compute cut fractionality f0
        f0 = scip.frac(primsol)

        # Rhs of the cut is the fractional part of the LP solution for the basic variable
        splitrhs = -f0

        #  Generate cut coefficients for the original variables
        for c in range(len(cols)):
            col = cols[c]
            assert col is not None
            if not col.isIntegral():
                continue

            lp_pos = col.getLPPos()

            status = col.getBasisStatus()

            # Get simplex tableau coefficient
            if lp_pos == lp_pos_col:
                rowelem = 1
            elif status == "lower":
                # Take coefficient if nonbasic at lower bound
                rowelem = binvarow[c]
            elif status == "upper":
                # Flip coefficient if nonbasic at upper bound: x --> u - x
                rowelem = -binvarow[c]
            else:
                # variable is nonbasic free at zero -> cut coefficient is zero, skip OR
                # variable is basic, skip
                assert status == "zero" or status == "basic"
                continue

            # Cut is defined when variables are in [0, infty). Translate to general bounds
            if not scip.isZero(rowelem):
                if col.getBasisStatus() == "upper":
                    rowelem = -rowelem
                    splitrhs += rowelem * col.getUb()
            else:
                splitrhs += rowelem * col.getLb()

            # Now floor or ceiling the element of the row
            if rowelem > f0:
                rowelem = math.ceil(rowelem)
            else:
                rowelem = math.floor(rowelem)

            # Add coefficient to cut in dense form
            splitcoefs[col.getLPPos()] = rowelem

        # Generate cut coefficients for the slack variables; skip basic ones
        for c in range(len(rows)):
            row = rows[c]
            assert row != None

            if not row.isIntegral() or row.isModifiable():
                continue

            status = row.getBasisStatus()

            # free slack variable shouldn't appear
            assert status != "zero"

            # Get simplex tableau coefficient
            if status == "lower":
                # Take coefficient if nonbasic at lower bound
                rowelem = binvrow[row.getLPPos()]
                # But if this is a >= or ranged constraint at the lower bound, we have to flip the row element
                if not scip.isInfinity(-row.getLhs()):
                    rowelem = -rowelem
            elif status == "upper":
                # Take element if nonbasic at upper bound - see notes at beginning of file: only nonpositive slack variables
                # can be nonbasic at upper, therefore they should be flipped twice and we can take the element directly.
                rowelem = binvrow[row.getLPPos()]
            else:
                assert status == "basic"
                continue

            # Cut is defined on original variables, so we replace slack by its definition
            if not scip.isZero(rowelem):
                # get lhs/rhs
                rlhs = row.getLhs()
                rrhs = row.getRhs()
                assert scip.isLE(rlhs, rrhs)
                assert not scip.isInfinity(rlhs) or not scip.isInfinity(rrhs)

                # If the slack variable is fixed, we can ignore this cut coefficient
                if scip.isFeasZero(rrhs - rlhs):
                    continue

                # Unflip slack variable and adjust rhs if necessary: row at lower means the slack variable is at its upper bound.
                # Since SCIP adds +1 slacks, this can only happen when constraints have a finite lhs
                if row.getBasisStatus() == "lower":
                    assert not scip.isInfinity(-rlhs)
                    rowelem = -rowelem

                rowcols = row.getCols()
                rowvals = row.getVals()

                assert len(rowcols) == len(rowvals)

                # Move the slack to its bounds
                act = scip.getRowLPActivity(row)
                rhsslack = rrhs - act
                if scip.isFeasZero(rhsslack):
                    assert row.getBasisStatus() == "upper"  # cutelem != 0 and row active at upper bound -> slack at lower, row at upper
                    splitrhs -= rowelem * (rrhs - row.getConstant())
                else:
                    assert scip.isFeasZero(act - rlhs)
                    splitrhs -= rowelem * (rlhs - row.getConstant())

                # Now floor or ceiling the element of the row
                if rowelem > f0:
                    rowelem = math.ceil(rowelem)
                else:
                    rowelem = math.floor(rowelem)

                # Eliminate slack variable: rowcols is sorted: [columns in LP, columns not in LP]
                for i in range(row.getNLPNonz()):
                    splitcoefs[rowcols[i].getLPPos()] -= rowelem * rowvals[i]

        return splitcoefs, math.floor(splitrhs)

    def branchexeclp(self, allowaddcons):

        # Get the branching candidates. Only consider the number of priority candidates (they are sorted to be first)
        # The implicit integer candidates in general shouldn't be branched on. Unless specified by the user
        # npriocands and ncands are the same (npriocands are variables that have been designated as priorities)
        branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.scip.getLPBranchCands()

        # Initialise scores for each variable
        scores = [-self.scip.infinity() for _ in range(npriocands)]

        # Get the basis indices information
        cols = self.scip.getLPColsData()
        scip_vars = [col.getVar() for col in cols]
        rows = self.scip.getLPRowsData()
        n_cols = self.scip.getNLPCols()
        n_rows = self.scip.getNLPRows()
        basis_var_to_tableau_row = [-1 for _ in range(n_cols)]
        basis_ind = self.scip.getLPBasisInd()

        for i in range(n_rows):
            if basis_ind[i] >= 0:
                basis_var_to_tableau_row[basis_ind[i]] = i

        # Start branching on splits that induce GMI cuts
        for i in range(npriocands):
            col = branch_cands[i].getCol()
            lp_pos = col.getLPPos()

            # Get the row of B^-1 for this basic integer variable with fractional solution value
            binvrow = self.scip.getLPBInvRow(lp_pos)

            # Get the tableau row for this basic integer variable with fractional solution value
            binvarow = self.scip.getLPBInvARow(lp_pos)

            # Get the GMI split
            split_coefs, split_rhs = self.getGMISplitFromRow(cols, rows, binvrow, binvarow, col.getPrimsol(), lp_pos)

            # Create the two children nodes
            child_1 = self.scip.createChild(1, 1)
            child_2 = self.scip.createChild(1, 1)
            split = 0
            for j, scip_var in enumerate(scip_vars):
                split += split_coefs[j] * scip_var
            cons_1 = self.scip.createConsFromExpr(split <= split_rhs, local=True)
            self.scip.addConsNode(child_1, cons_1)
            cons_2 = self.scip.createConsFromExpr(split >= split_rhs + 1, local=True)
            self.scip.addConsNode(child_2, cons_2)

            return {"result": SCIP_RESULT.BRANCHED}


def create_model():
    scip = Model()
    # Disable separating and heuristics as we want to branch on the problem many times before reaching optimality.
    scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    scip.setSeparating(SCIP_PARAMSETTING.OFF)
    scip.setLongintParam("limits/nodes", 2000)

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
        scip.addCons(quicksum(v for v in more_vars[50::2]) <= (40 - i) * quicksum(v for v in more_vars[200::2]))

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


def test_gmi_branching():
    scip = create_model()

    gmi_branch_rule = GMIBranchingRule(scip)
    scip.includeBranchrule(gmi_branch_rule, "gmi branch rule", "custom gmi branching rule",
                           priority=10000000, maxdepth=-1, maxbounddist=1)

    scip.optimize()
    if scip.getStatus() == "optimal":
        assert scip.isEQ(-112196, scip.getObjVal())
    else:
        assert -112196 <= scip.getObjVal()
