from typing_extensions import assert_type

from pyscipopt import (
    LP,
    SCIP_BASESTAT,
    SCIP_EVENTTYPE,
    SCIP_HEURTIMING,
    SCIP_IMPLINTTYPE,
    SCIP_LPSOLSTAT,
    SCIP_NODETYPE,
    SCIP_PARAMSETTING,
    SCIP_RESULT,
    SCIP_ROWORIGINTYPE,
    SCIP_SOLORIGIN,
    SCIP_STAGE,
    Eventhdlr,
    Model,
    Variable,
)
from pyscipopt.scip import Event, Node, Row, Solution


def configure(model: Model, setting: SCIP_PARAMSETTING) -> None:
    model.setPresolve(setting)
    model.setHeuristics(setting)
    model.setSeparating(setting)


def check_getters(
    model: Model,
    event: Event,
    node: Node,
    row: Row,
    sol: Solution,
    var: Variable,
    lp: LP,
) -> None:
    assert_type(model.getStage(), SCIP_STAGE)
    assert_type(model.getHeurTiming("rounding"), SCIP_HEURTIMING)
    assert_type(model.getLPSolstat(), SCIP_LPSOLSTAT)
    assert_type(model.branchLPExact(), SCIP_RESULT)
    assert_type(event.getType(), SCIP_EVENTTYPE)
    assert_type(node.getType(), SCIP_NODETYPE)
    assert_type(row.getOrigintype(), SCIP_ROWORIGINTYPE)
    assert_type(sol.getOrigin(), SCIP_SOLORIGIN)
    assert_type(var.getImplType(), SCIP_IMPLINTTYPE)
    assert_type(lp.getBase(), tuple[list[SCIP_BASESTAT], list[SCIP_BASESTAT]])


model = Model()
configure(model, SCIP_PARAMSETTING.OFF)
configure(model, SCIP_PARAMSETTING(3))
configure(model, 3)
configure(model, SCIP_RESULT.FEASIBLE)
model.setPresolve(3)
model.setPresolve(SCIP_RESULT.FEASIBLE)

mask = SCIP_EVENTTYPE.NODEFOCUSED | SCIP_EVENTTYPE.LPSOLVED
assert_type(mask, SCIP_EVENTTYPE)
handler = Eventhdlr()
model.catchEvent(mask, handler)
model.dropEvent(mask, handler)
model.catchEvent(SCIP_PARAMSETTING.OFF, handler)
