import json
import sqlite3
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.portfolio_tracker.laura_case import save_case_section
from src.portfolio_tracker.strategy_store import load_client_mandate
from src.simulation.laura_funding import LauraPolicy
from ui.laura_plan import build_funding_run


def _allocation():
    return pd.DataFrame([
        {"Asset": "GROWTH", "2027–2030 %": 100., "2031–2032 %": 40., "2033–2036 %": 0., "2037–2041 %": 0.},
        {"Asset": "RESERVE", "2027–2030 %": 0., "2031–2032 %": 60., "2033–2036 %": 100., "2037–2041 %": 100.},
    ])


def test_run_exports_reproducible_paired_inputs_and_separate_conditional_state():
    annual = pd.DataFrame({"GROWTH": [.2, -.1, .04, .05], "RESERVE": [.03, .04, -.02, .01]}, index=[2019, 2020, 2021, 2022])
    first = build_funding_run(annual, _allocation(), LauraPolicy(), opening_2031=600_000,
                             state_description="Illustrative 2031 state", n_scenarios=100,
                             candidates={"Constant growth": [1., 0.]})
    second = build_funding_run(annual, _allocation().iloc[::-1], LauraPolicy(), opening_2031=600_000,
                              state_description="Illustrative 2031 state", n_scenarios=100)
    np.testing.assert_allclose(first["projection"].portfolio_2033, second["projection"].portfolio_2033)
    assert len(first["stress_rows"]) == 7
    assert len(first["comparison_rows"]) == 2
    assert first["metadata"]["history_years"] == [2019, 2020, 2021, 2022]
    assert first["interval"]["valuation_date"] == "2031-01-01"
    assert "not a binding commitment" in first["fundraising_draft"]
    json.dumps(first["metadata"], allow_nan=False)
    json.dumps(first["stress_rows"], allow_nan=False)
    json.dumps(first["projection"].scenario_rows(), allow_nan=False)
    for j in range(100):
        p = first["projection"]
        assert p.portfolio_2033[j] == pytest.approx(p.allocated_reserve[j] + p.facility[j] + p.flexibility[j])


def _app(database):
    return AppTest.from_string(dedent(f"""
        import sqlite3
        from contextlib import contextmanager
        import numpy as np
        import pandas as pd
        from ui.laura_plan import render_laura_plan
        @contextmanager
        def connection():
            with sqlite3.connect({str(database)!r}) as conn:
                yield conn
        index = pd.bdate_range('2019-01-01', '2023-12-31')
        result = {{'tickers': ['GROWTH', 'RESERVE'], 'weights': np.array([.6, .4]),
                  'returns': pd.DataFrame({{'GROWTH': np.full(len(index), .0003),
                                            'RESERVE': np.full(len(index), .0001)}}, index=index)}}
        render_laura_plan({{'username': 'team'}}, result, connection)
    """), default_timeout=30)


def test_client_plan_renders_case_submissions_and_saves_verbatim_drafts(tmp_path):
    database = tmp_path / "client.db"
    app = _app(database).run()
    assert not app.exception
    assert app.dataframe[0].value["Deposit USD"].sum() == 450_000
    app.radio[0].set_value("Submissions").run()
    assert not app.exception
    original = "Exact note\nwith original punctuation: A&B."
    app.text_input(key="laura_trade_ref_0").set_value("execution-1")
    app.text_area(key="laura_trade_note_0").set_value(original)
    app.text_area(key="laura_reflection_0").set_value("word " * 101)
    app.button[0].click().run()
    assert not app.exception
    with sqlite3.connect(database) as conn:
        saved = load_client_mandate(conn)["payload"]["laura_case"]["deliverables"]
    assert saved["notes"][0]["note"] == original
    assert not app.dataframe[1].value["passed"].all()


def test_funding_form_runs_saves_and_renders_results_without_touching_other_sections(tmp_path):
    database = tmp_path / "funding.db"
    with sqlite3.connect(database) as conn:
        save_case_section(conn, "planning", {"allocation": _allocation().to_dict(orient="records"),
                                             "rationale": "Model reserve with the explicit reserve proxy; staged risk reduction."}, updated_by="team")
        save_case_section(conn, "deliverables", {"pitch": "Keep this draft"}, updated_by="team")
    app = _app(database).run()
    app.radio[0].set_value("Funding Model").run()
    assert not app.exception
    app.checkbox[0].check()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 6
    assert any("5th–95th" in m.value for m in app.markdown)
    with sqlite3.connect(database) as conn:
        saved = load_client_mandate(conn)["payload"]["laura_case"]
    assert saved["planning"]["last_run"]["metadata"]["scenario_count"] == 10_000
    assert saved["deliverables"]["pitch"] == "Keep this draft"
