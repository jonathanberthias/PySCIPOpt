import json
import pickle
from enum import IntEnum, IntFlag

import pytest

import pyscipopt
from pyscipopt import (
    SCIP_EVENTTYPE,
    SCIP_HEURTIMING,
    SCIP_IMPLINTTYPE,
    SCIP_LPSOLSTAT,
    SCIP_NODETYPE,
    SCIP_PARAMSETTING,
    SCIP_ROWORIGINTYPE,
    SCIP_SOLORIGIN,
    SCIP_STAGE,
    Model,
)


@pytest.mark.parametrize(
    "enum",
    [getattr(pyscipopt, name) for name in dir(pyscipopt) if name.startswith("SCIP_")],
)
def test_enum_round_trip(enum):
    for name, member in enum.__members__.items():
        assert isinstance(member, (IntEnum, IntFlag))
        assert enum[name] is member
        assert enum(member.value) is member
        assert pickle.loads(pickle.dumps(member)) is member
        assert json.loads(json.dumps(member)) == member.value


def test_enum_parameter_settings():
    model = Model()
    model.hideOutput()
    for setting in SCIP_PARAMSETTING:
        for setter in (model.setPresolve, model.setHeuristics, model.setSeparating):
            setter(setting)
            setter(int(setting))

    assert model.getStage() is SCIP_STAGE.PROBLEM
    assert model.getStage().name == "PROBLEM"
    assert model.getStageName() == "PROBLEM"


def test_variable_and_solution_enum_getters():
    model = Model()
    var = model.addVar()
    assert var.getImplType() is SCIP_IMPLINTTYPE.NONE

    sol = model.createOrigSol()
    assert sol.getOrigin() is SCIP_SOLORIGIN.ORIGINAL
    model.freeSol(sol)


def test_combined_heuristic_timing():
    model = Model()
    timing = SCIP_HEURTIMING.BEFORENODE | SCIP_HEURTIMING.AFTERLPNODE
    assert isinstance(timing, SCIP_HEURTIMING)
    model.setHeurTiming("rounding", timing)
    returned_timing = model.getHeurTiming("rounding")
    assert isinstance(returned_timing, SCIP_HEURTIMING)
    assert returned_timing == timing


def test_combined_event_mask():
    model = Model()
    model.hideOutput()
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    x = model.addVar(vtype="B", obj=1)
    model.addCons(x >= 1)

    mask = SCIP_EVENTTYPE.NODEFOCUSED | SCIP_EVENTTYPE.BESTSOLFOUND
    assert isinstance(mask, SCIP_EVENTTYPE)
    seen = []

    def callback(model, event):
        eventtype = event.getType()
        assert isinstance(eventtype, SCIP_EVENTTYPE)
        assert eventtype & mask
        assert event.getName() == eventtype.name
        seen.append(eventtype)

    model.attachEventHandlerCallback(callback, [mask])
    model.optimize()
    assert SCIP_EVENTTYPE.BESTSOLFOUND in seen
    assert model.getStatus() == "optimal"


def test_enum_getters_during_lp_solve():
    model = Model()
    model.hideOutput()
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    x = model.addVar(obj=1)
    y = model.addVar(obj=2)
    model.addCons(x + y >= 1)
    seen = []

    def callback(model, event):
        row = model.getLPRowsData()[0]
        seen.append(
            (
                event.getType(),
                model.getStage(),
                model.getCurrentNode().getType(),
                model.getLPSolstat(),
                row.getOrigintype(),
            )
        )

    model.attachEventHandlerCallback(callback, [SCIP_EVENTTYPE.FIRSTLPSOLVED])
    model.optimize()
    assert seen
    expected = (
        SCIP_EVENTTYPE.FIRSTLPSOLVED,
        SCIP_STAGE.SOLVING,
        SCIP_NODETYPE.FOCUSNODE,
        SCIP_LPSOLSTAT.OPTIMAL,
        SCIP_ROWORIGINTYPE.CONS,
    )
    for values in seen:
        assert all(value is member for value, member in zip(values, expected))
