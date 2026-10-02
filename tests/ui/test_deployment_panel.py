import sqlite3

from streamlit.testing.v1 import AppTest


APP = '''
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
from src.analytics.entry_quality import calculate_entry_quality
from ui.deployment import render_deployment_planner
p = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(4).normal(.001, .01, 900))),
    index=pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=900))
entry = calculate_entry_quality('TEST', p, p, target_weight=.05)
render_deployment_planner('TEST', entry, p, p, key_prefix='test',
    connection_factory=lambda: sqlite3.connect(st.session_state['test_db']),
    scope='test_team', actor='analyst', can_edit=st.session_state.get('can_edit', True))
'''

APP_WITH_ZERO_TARGET = '''
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
from src.analytics.entry_quality import calculate_entry_quality
from ui.deployment import render_deployment_planner
p = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(4).normal(.001, .01, 900))),
    index=pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=900))
entry = calculate_entry_quality('TEST', p, p, target_weight=0.)
render_deployment_planner('TEST', entry, p, p, key_prefix='zero',
    connection_factory=lambda: sqlite3.connect(st.session_state['test_db']),
    scope='test_team', actor='analyst')
'''


def widget(app, kind, label):
    return next(w for w in getattr(app, kind) if w.label == label)


def open_app(tmp_path):
    app = AppTest.from_string(APP, default_timeout=30)
    app.session_state['test_db'] = str(tmp_path / 'deployment.db')
    app.run()
    widget(app, 'checkbox', 'Open deployment planner').check().run()
    return app


def create_plan(app):
    widget(app, 'text_area', 'Approved investment thesis').set_value('Durable competitive position')
    widget(app, 'text_area', 'Conditions that invalidate the thesis').set_value('Persistent margin deterioration')
    widget(app, 'text_input', 'Investment horizon').set_value('Five years')
    widget(app, 'text_input', 'Mandate / approval reference').set_value('Committee decision 1')
    widget(app, 'button', 'Save approved plan').click().run()
    assert not app.exception
    assert widget(app, 'selectbox', 'Deployment plan').value != 'New plan'


def test_save_review_execution_and_audit(tmp_path):
    app = open_app(tmp_path)
    create_plan(app)
    widget(app, 'text_input', 'Portfolio snapshot / reconciliation reference').set_value('WInS verified 1')
    widget(app, 'checkbox', 'Thesis remains valid').check()
    widget(app, 'checkbox', 'Portfolio limits, liquidity, cash for fees and mandate checks passed').check().run()
    assert not app.exception
    assert next(m for m in app.metric if m.label == 'Proposed purchase').value == '5,000.00'
    widget(app, 'button', 'Record review').click().run()
    assert not app.exception
    widget(app, 'text_input', 'Unique official execution reference').set_value('trade-001')
    widget(app, 'number_input', 'Actual executed notional').set_value(5000.)
    widget(app, 'button', 'Record execution evidence').click().run()
    assert not app.exception
    assert not app.error
    with sqlite3.connect(tmp_path / 'deployment.db') as conn:
        assert conn.execute('SELECT COUNT(*) FROM deployment_events').fetchone()[0] == 2
    assert len(app.get('download_button')) >= 2


def test_compare_and_export(tmp_path):
    app = open_app(tmp_path)
    widget(app, 'checkbox', 'Include repeated historical windows').check()
    widget(app, 'button', 'Run policy comparison').click().run(timeout=60)
    assert not app.exception
    assert len(app.dataframe) == 2
    assert any(w.label == 'Download comparison evidence' for w in app.get('download_button'))


def test_read_only_cannot_save(tmp_path):
    app = open_app(tmp_path)
    app.session_state['can_edit'] = False
    app.run()
    assert widget(app, 'button', 'Save approved plan').disabled


def test_timing_window_and_interactive_export(tmp_path, monkeypatch):
    import pandas as pd
    from src.analytics.deployment_timing import TimingEstimate
    from ui import deployment

    estimate = TimingEstimate(str(pd.Timestamp.now(tz='UTC').date()), 'estimated', 'Synthetic test condition',
        sample_count=12, probabilities={'5': .25, '10': .5, '20': .75}, no_hit_probability=.25, median_sessions=10)
    monkeypatch.setattr(deployment, '_timing_estimate', lambda *args: estimate)
    app = open_app(tmp_path)
    widget(app, 'checkbox', 'Research mode — unvalidated sizing and timing').check().run()
    create_plan(app)
    widget(app, 'checkbox', 'Estimate next entry window from historical analogues').check().run()
    assert not app.exception
    assert next(m for m in app.metric if m.label == 'Conditions within 5 sessions').value == '25%'
    assert len(app.get('plotly_chart')) == 2
    assert any(w.label == 'Download interactive purchase timeline' for w in app.get('download_button'))
    assert any(w.label == 'Download timing evidence' for w in app.get('download_button'))
    assert any(w.label == 'Download entry-execution report evidence' for w in app.get('download_button'))


def test_competition_view_has_no_timing_or_score_adjusted_policy(tmp_path):
    app = open_app(tmp_path)
    assert widget(app, 'selectbox', 'Purchase policy').options == ['Immediate', 'Fixed schedule']
    create_plan(app)
    assert not any('Estimate next entry window' in w.label for w in app.checkbox)
    assert any(w.label == 'Download competition evidence working paper' for w in app.get('download_button'))
    assert any('Missing:' in c.value for c in app.caption)


def test_two_stage_pattern_shows_a_fixed_second_half_review(tmp_path):
    app = open_app(tmp_path)
    widget(app, 'selectbox', 'Deployment pattern').set_value('Two-stage: 50% now / 50% later').run()
    assert widget(app, 'number_input', 'Days until second-half review').value == 14
    create_plan(app)
    assert any('Two-stage plan:' in info.value for info in app.info)
    assert any(row['stage'] == 'Second half (50%) — completion review'
               for row in app.dataframe[0].value.to_dict('records'))


def test_research_mode_offers_the_signal_gated_entry_policy(tmp_path):
    app = open_app(tmp_path)
    widget(app, 'checkbox', 'Research mode — unvalidated sizing and timing').check().run()
    widget(app, 'selectbox', 'Purchase policy').set_value('Signal-gated').run()
    assert widget(app, 'number_input', 'Business-day proxy between model reviews').value == 5
    assert widget(app, 'number_input', 'Model review count including completion').value == 5


def test_planner_recovers_from_an_absent_upstream_target_weight(tmp_path):
    app = AppTest.from_string(APP_WITH_ZERO_TARGET, default_timeout=30)
    app.session_state['test_db'] = str(tmp_path / 'deployment.db')
    app.run()
    widget(app, 'checkbox', 'Open deployment planner').check().run()
    assert not app.exception
    assert not app.error
    assert widget(app, 'number_input', 'Target portfolio weight (%)').value == 5.
