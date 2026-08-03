# Client Behavioral Profile

The **Strategy & Decisions -> Behavioral Profile** tab documents how a client
is likely to make investment decisions under uncertainty and converts the
assessment into explicit governance rules. The profile is versioned inside the
shared Client Mandate.

It is an investment-governance aid, not a clinical diagnosis, personality test,
or validated psychometric instrument. The score must never be presented as a
medical fact, an official Wharton score, or a substitute for a client
conversation.

## Evidence status

Every assessment records whether its inputs come from:

- a client interview;
- observed client behavior;
- an adviser assessment;
- an analyst assumption; or
- unverified information.

Analyst assumptions and unverified profiles remain visibly flagged. The
evidence-reference field should point to reviewable interview notes, observed
decisions, or the official case. Do not invent answers to make the profile look
complete.

Behavioral information can be sensitive. Store only information needed for the
investment process, follow the team's access and retention rules, and avoid
collecting protected characteristics or unrelated personal details.

## Questionnaire convention

The questionnaire uses twelve statements scored from 1 (strongly disagree) to
5 (strongly agree). Every statement is intentionally oriented in the same
direction: a higher answer indicates greater potential vulnerability.

The categories are:

- loss aversion;
- overconfidence;
- recency bias;
- herding and social proof;
- anchoring;
- the disposition effect; and
- action bias.

Each answer is transformed linearly onto a 0-100 scale:

`score = (answer - 1) / 4 * 100`

Category scores are simple averages of their answered items. The overall score
is the average of every answered question. No hidden weighting, machine-learning
model, or external personal data is used.

Score bands are:

- below 25: Low;
- 25 to below 50: Moderate;
- 50 to below 70: Elevated;
- 70 or higher: High.

A high score is a prompt for process protection, not a judgement about the
client's intelligence or character.

## Drawdown behavior

The client selects an intended response at portfolio drawdowns of 10%, 20%, and
30%. Responses are weighted more heavily at deeper drawdowns. Rebalancing to
the existing policy target receives the highest discipline score. Selling all
risk assets receives the lowest. Adding risk beyond policy targets is not
treated as disciplined merely because it is aggressive.

The resulting drawdown-discipline score is compared with the declared risk
tolerance in the Client Mandate. When intended behavior is materially less
resilient than the declared tolerance, the system flags the mismatch and
recommends using the lower effective tolerance until the inconsistency is
resolved.

Risk willingness, financial risk capacity, required return, and behavioral
discipline are different concepts. None should be used as a substitute for the
others.

## Guardrails

Scores at or above 50 generate category-specific controls. Examples include:

- a pre-committed response at each drawdown threshold;
- a 48-hour cooling-off period for unscheduled trades;
- independent challenge and disconfirming evidence before concentration;
- use of full-cycle base rates before extrapolating recent returns;
- re-underwriting fair value without reference to purchase price;
- symmetric forward-looking sell rules for winners and losers; and
- an explicit no-trade outcome at scheduled decision meetings.

The profile also records decision makers, trusted sources, communication style,
communication frequency, known stress triggers, and the agreed escalation
protocol. These inputs turn the score into an operating plan.

## Review policy

Review the profile when:

1. actual behavior during market stress differs from the recorded plan;
2. the mandate, goals, liquidity needs, or decision makers change;
3. a material life or organizational event changes risk capacity;
4. the client gains relevant experience with losses; or
5. the evidence status can be upgraded from assumption to confirmed input.

Actual observed behavior should be documented without hindsight bias. A client
can revise preferences, but changes made during stress should not automatically
override the pre-agreed policy without the recorded escalation process.
