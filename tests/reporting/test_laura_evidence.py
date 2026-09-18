from copy import deepcopy

import pandas as pd
import pytest

from src.reporting.evidence_studio import create_report_workspace, register_client_plan_evidence
from src.simulation.laura_funding import LauraPolicy
from ui.laura_plan import build_funding_run


def test_client_calculation_becomes_immutable_citable_report_evidence():
    history = pd.DataFrame({"ASSET": [.03, .04, .05]}, index=[2020, 2021, 2022])
    allocation = pd.DataFrame([{"Asset": "ASSET", "2027–2030 %": 100., "2031–2032 %": 100.,
                                "2033–2036 %": 100., "2037–2041 %": 100.}])
    run = build_funding_run(history, allocation, LauraPolicy(), opening_2031=550_000,
                           state_description="Modeled state", n_scenarios=20)
    planning = {"rationale": "Team rationale", "last_run": {"metadata": run["metadata"],
                 "summary": run["projection"].summary(), "partner_interval": run["interval"]}}
    workspace = create_report_workspace("laura", "final", "Final working report", created_by="team")
    attached = register_client_plan_evidence(workspace, planning, verified_by="reviewer")
    assert workspace["evidence"] == {}
    record = next(iter(attached["evidence"].values()))
    assert record["source_type"] == "model_output"
    assert record["verified_by"] == "reviewer"
    assert record["client_plan"]["metadata"]["history_years"] == [2020, 2021, 2022]
    planning["last_run"]["summary"]["all_payments_probability"] = -1
    assert record["client_plan"]["summary"]["all_payments_probability"] >= 0
    frozen = deepcopy(workspace)
    frozen["status"] = "frozen"
    with pytest.raises(ValueError, match="draft"):
        register_client_plan_evidence(frozen, planning, verified_by="reviewer")


def test_missing_model_run_is_not_accepted_as_evidence():
    workspace = create_report_workspace("laura", "final", "Final working report", created_by="team")
    with pytest.raises(ValueError, match="complete"):
        register_client_plan_evidence(workspace, {}, verified_by="reviewer")
