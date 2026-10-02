"""Calendar-anchored deployment proposals. No order submission or price forecasts."""
from dataclasses import asdict, dataclass, field
from math import isfinite

import pandas as pd

from src.analytics.entry_quality import EntryQualityResult


POLICIES = ('Immediate', 'Fixed schedule', 'Score-adjusted', 'Signal-gated')
POLICY_VERSION = '1.0'


def day(value):
    result = pd.Timestamp(value).tz_localize(None).normalize()
    if pd.isna(result):
        raise ValueError('A valid date is required.')
    return result


def nonnegative(value, name):
    if isinstance(value, bool) or not isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative.')


@dataclass(frozen=True)
class DeploymentPlan:
    ticker: str
    benchmark: str
    target_weight: float
    budget: float
    start_date: str
    reviews: int = 5
    interval_days: int = 7
    policy: str = 'Fixed schedule'
    thesis: str = ''
    invalidation: str = ''
    horizon: str = ''
    mandate_reference: str = ''
    max_position_weight: float = 1.
    max_purchase: float = 1_000_000_000.
    adjustment: float = .5
    cost_bps: float = 10.
    version: str = POLICY_VERSION
    competition_context: dict[str, str] = field(default_factory=dict)

    def validate(self):
        if not self.ticker.strip() or not self.benchmark.strip():
            raise ValueError('Security and benchmark are required.')
        for name in ('target_weight', 'budget', 'max_position_weight', 'max_purchase', 'adjustment', 'cost_bps'):
            nonnegative(getattr(self, name), name)
        if not 0 < self.target_weight <= self.max_position_weight <= 1:
            raise ValueError('Target must be positive and within the position limit (at most 100%).')
        if self.budget <= 0 or self.max_purchase <= 0 or self.adjustment > .5:
            raise ValueError('Positive budget/purchase limit and adjustment between 0 and 0.5 required.')
        if self.cost_bps > 1000:
            raise ValueError('Execution cost assumption must be at most 1000 basis points.')
        if (type(self.reviews) is not int or not 1 <= self.reviews <= 52 or
                type(self.interval_days) is not int or not 1 <= self.interval_days <= 365):
            raise ValueError('Use 1–52 reviews and a positive interval of at most 365 days.')
        if self.policy not in POLICIES or self.version != POLICY_VERSION:
            raise ValueError('Unsupported policy or version.')
        day(self.start_date)
        if not isinstance(self.competition_context, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in self.competition_context.items()):
            raise ValueError('Competition context must contain text fields.')
        if not all(x.strip() for x in (self.thesis, self.invalidation, self.horizon, self.mandate_reference)):
            raise ValueError('Thesis, invalidation conditions, horizon and mandate reference are required.')

    def dates(self):
        self.validate()
        # Signal-gated research plans use an explicit business-day proxy so their saved
        # review cadence matches the walk-forward model. Other plans remain calendar based.
        if self.policy == 'Signal-gated':
            return [day(self.start_date) + pd.offsets.BDay(i * self.interval_days)
                    for i in range(self.reviews)]
        # Calendar dates are review dates; execution uses the next observed market session.
        return [day(self.start_date) + pd.Timedelta(days=i * self.interval_days)
                for i in range(self.reviews)]

    def is_two_stage(self):
        """Whether the fixed calendar represents a 50/50 initial/completion plan."""
        return self.policy == 'Fixed schedule' and self.reviews == 2

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class DeploymentState:
    portfolio_value: float
    holding_value: float
    cash: float
    spent: float = 0.
    pending_position: float = 0.
    pending_plan: float = 0.
    reserved_cash: float = 0.
    thesis_valid: bool = True
    constraints_valid: bool = True
    snapshot_date: str = ''
    snapshot_reference: str = ''
    last_review_date: str | None = None

    def validate(self):
        for name in ('portfolio_value', 'holding_value', 'cash', 'spent',
                     'pending_position', 'pending_plan', 'reserved_cash'):
            nonnegative(getattr(self, name), name)
        if self.portfolio_value <= 0 or self.holding_value + self.cash > self.portfolio_value + .01:
            raise ValueError('Holdings and cash cannot exceed positive portfolio value.')
        if not self.pending_plan <= self.pending_position <= self.reserved_cash <= self.cash:
            raise ValueError('Pending plan ≤ pending security ≤ total reserved cash ≤ cash is required.')
        if type(self.thesis_valid) is not bool or type(self.constraints_valid) is not bool:
            raise ValueError('Thesis/constraint checks must be explicit booleans.')
        day(self.snapshot_date)
        if not self.snapshot_reference.strip():
            raise ValueError('A portfolio snapshot reference is required.')


@dataclass(frozen=True)
class DeploymentDecision:
    as_of: str
    scheduled_review: str | None
    next_review: str | None
    status: str
    allocation_gap: float
    remaining_budget: float
    baseline_purchase: float
    proposed_purchase: float
    cumulative_policy_target: float
    estimated_cost: float
    explanation: str

    def to_dict(self):
        return asdict(self)


def propose_deployment(plan: DeploymentPlan, state: DeploymentState, *, as_of,
                       entry: EntryQualityResult | None = None, signal=None) -> DeploymentDecision:
    plan.validate()
    state.validate()
    today = day(as_of)
    dates = plan.dates()
    eligible = [d for d in dates if d <= today]
    due = eligible[-1] if eligible else None
    future = next((str(d.date()) for d in dates if d > today), None)
    committed = state.spent + state.pending_plan
    remaining = max(0., plan.budget - committed)
    gap = max(0., plan.target_weight * state.portfolio_value - state.holding_value - state.pending_position)
    scheduled = min(plan.budget, len(eligible) * plan.budget / plan.reviews)
    baseline = max(0., scheduled - committed)
    target = scheduled
    status, reason = 'Proposed', 'Calendar-anchored purchase.'
    blocked = False
    if today != day(state.snapshot_date):
        status, reason, blocked = 'Review required', 'Refresh the portfolio snapshot for this review date.', True
    elif not state.thesis_valid or not state.constraints_valid:
        status, reason, blocked = 'Review required', 'Thesis or portfolio constraints have not passed review.', True
    elif due is None:
        status, reason, blocked = 'Not due', 'The first scheduled review has not arrived.', True
    elif state.last_review_date and day(state.last_review_date) >= due:
        status, reason, blocked = 'Already reviewed', 'This scheduled review is already recorded.', True
    elif remaining <= .01 or gap <= .01:
        status, reason, blocked = 'Funded', 'No uncommitted budget or target allocation gap remains.', True
    elif plan.policy == 'Immediate':
        target = plan.budget
        reason = 'Deploy the approved remaining budget, subject to current limits.'
    elif plan.is_two_stage():
        if due == dates[0]:
            reason = ('Two-stage plan: propose the first half now. The remaining half has a '
                      f'predefined review on {dates[1].date()}.')
        else:
            reason = ('Two-stage completion review: propose the remaining approved budget, '
                      'subject to current limits.')
    elif plan.policy == 'Signal-gated':
        if due == dates[0]:
            target = plan.budget / 2
            reason = ('Signal-gated plan: propose the approved first half. The remaining half is '
                      'reviewed by the saved model schedule and forced at the final review.')
        elif due == dates[-1]:
            target = plan.budget
            reason = 'Signal-gated completion review: propose the remaining approved budget within current limits.'
        elif (signal is None or getattr(signal, 'as_of', None) != str(today.date()) or
              getattr(signal, 'status', None) != 'research_ready'):
            status, reason, blocked = 'Review required', (
                'A same-date, research-ready entry-algorithm result is required before deferring or completing the remaining tranche.'), True
        elif getattr(signal, 'action', '') == 'Buy remaining tranche now':
            target = plan.budget
            reason = ('Signal-gated review qualified: propose the remaining approved budget within current limits.')
        else:
            target = committed
            reason = ('Signal-gated review deferred the remaining tranche. Keep the next saved review and '
                      'the final completion review; do not infer a price forecast.')
    elif plan.policy == 'Score-adjusted':
        if (entry is None or entry.entry_score is None or not entry.data_quality.get('complete') or
                entry.ticker.upper() != plan.ticker.upper() or entry.benchmark.upper() != plan.benchmark.upper() or
                day(entry.as_of) != today):
            status, reason, blocked = 'Review required', 'A complete matching Entry Quality assessment is required for this date.', True
        elif due == dates[-1]:
            target = plan.budget
            reason = 'Completion review: deploy the remaining approved budget within limits.'
        elif entry.entry_score >= 80:
            target = min(plan.budget, scheduled + plan.adjustment * plan.budget / plan.reviews)
            reason = 'Favorable conditions: bring forward up to half of one tranche.'
        elif entry.entry_score < 40:
            target = max(0., scheduled - plan.adjustment * plan.budget / plan.reviews)
            reason = 'Unfavorable conditions: defer up to half of one tranche.'
        else:
            reason = 'Maintain the calendar schedule; no score adjustment.'
    desired = max(0., target - committed)
    # Fees reduce portfolio equity; solve x <= w*(V-fee*x)-H-pending.
    purchase = 0. if blocked else min(desired, remaining, gap / (1 + plan.target_weight * plan.cost_bps / 10_000),
        (state.cash - state.reserved_cash) / (1 + plan.cost_bps / 10_000), plan.max_purchase)
    if not blocked and purchase + .01 < desired:
        status = 'Limited'
        reason += ' Purchase capped by budget, allocation gap, unreserved cash or purchase limit.'
    if due == dates[-1] and remaining - purchase > .01 and not blocked:
        reason += ' Unfunded remainder requires an explicit completion exception or a new plan.'
    return DeploymentDecision(str(today.date()), str(due.date()) if due is not None else None,
        future, status, gap, remaining, baseline, purchase, target, purchase * plan.cost_bps / 10_000, reason)
