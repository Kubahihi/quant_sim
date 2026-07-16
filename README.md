# Quant Investment Platform

![Lines of Code](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.codetabs.com%2Fv1%2Floc%3Fgithub%3DKubahihi%2Fquant_sim&query=%240.linesOfCode&label=Lines%20of%20Code&color=blue)

Streamlit aplikace pro vyhodnoceni investicniho portfolia:
- validace tickeru a vah
- nacitani trznich dat z Yahoo Finance
- metriky, score, flagy a rizikovy rozbor
- AI komentare pres Groq API (OpenAI kompatibilni klient)
- export vsech vysledku do vice-strankoveho PDF + CSV + JSON

## 1) Jak projekt spustit lokalne

### Vytvoreni virtualniho prostredi

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Instalace zavislosti

```bash
pip install -r requirements.txt
```

### Nastaveni GROQ_API_KEY

Aplikace nacita API klic v tomto poradi:
1. `st.secrets["GROQ_API_KEY"]`
2. environment variable `GROQ_API_KEY`

Moznost A: Streamlit secrets

Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Path .streamlit -Force | Out-Null
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

macOS/Linux:

```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Potom doplnte realny klic do `.streamlit/secrets.toml`.

Moznost B: Environment variable

Windows (PowerShell):

```powershell
$env:GROQ_API_KEY="gsk_..."
```

macOS/Linux:

```bash
export GROQ_API_KEY="gsk_..."
```

### Spusteni aplikace

```bash
streamlit run ui/streamlit_app.py
```

## 2) Deploy na Streamlit Cloud

1. Nahrajte repozitar do GitHubu.
2. Ve Streamlit Cloud zvolte repo a soubor `ui/streamlit_app.py`.
3. Do sekce **Secrets** vlozte:

```toml
GROQ_API_KEY="gsk_..."
```

4. Deploy.

Poznamka: Pokud klic chybi nebo Groq neodpovi, aplikace bezi dal a pouzije deterministic fallback komentar.

### Sdilena online databaze

Decision log, historie jeho uprav, data dashboardu a verzovane makro snapshoty
pro regionalni analyzu se ukladaji do sdilene Turso/libSQL databaze, jakmile
jsou nastaveny `TURSO_DATABASE_URL` a `TURSO_AUTH_TOKEN`. Aplikace pri startu
automaticky vytvori nebo zaktualizuje tabulky a po kazdem ulozeni synchronizuje
zmeny online. Makro snapshoty pro referencni rok 2024 maji sestihodinovou
expiraci; stejna data tak sdileji vsechny bezici instance a po expiraci se
automaticky obnovi z primarnich zdroju.

1. V Turso vytvorte databazi a vygenerujte pristupovy token.
2. Hodnoty vlozte do `.streamlit/secrets.toml` pri lokalnim spusteni nebo do
   **Secrets** v Streamlit Cloud pri nasazeni.
3. Nasadte aplikaci; vsichni clenove tymu pak pouziji stejny decision log i
   stejnou makro cache. Zadna dalsi databazova migrace ani secret nejsou potreba.

Bez techto dvou secrets aplikace zamerne pouzije pouze lokalni databazi, aby
neukladala data na neznamy vzdaleny server.

## 3) Zakladni logika aplikace

### Vstup
- tickery (1 na radek)
- vahy v % (1 na radek, nepovinne)
- datumovy rozsah
- risk-free rate
- risk profile (`conservative`, `balanced`, `aggressive`)
- horizont simulace + pocet Monte Carlo simulaci

### Validace
- prazdne tickery
- duplicitni tickery
- neplatne vahy
- nesoulad poctu tickeru a vah
- zaporne vahy
- soucet vah mimo 100 %

Pokud je soucet vah mimo 100 %, aplikace je normalizuje a upozorni uzivatele.

### Nacitani dat
- data se stahuji z Yahoo Finance
- market data jsou cachovana pomoci `st.cache_data` (TTL 1 hodina)
- chybejici tickery se oznaci, dostupna cast portfolia se prepocita

### Vypocitane metriky
- denni vynosy portfolia
- anualizovany vynos
- volatilita
- Sharpe ratio
- max drawdown
- korelacni matice
- koncentrace (HHI, effective holdings, max vaha)

## 4) Jak funguje scoring

Deterministic score (`0-100`) je zalozeny na pravidlech:
- vysoka koncentrace
- slaba diverzifikace
- vysoka volatilita
- nizke Sharpe ratio
- velky drawdown
- vysoka prumerna korelace

Kazde pravidlo pridava penalizaci. Vysledkem je:
- numericke score
- slovni rating
- seznam flagu
- fallback text pouzitelny i bez AI

## 5) Jak funguje Groq AI vrstva

Implementace je v `src/ai/ai_review.py`:
- klient: `from openai import OpenAI`
- base URL: `https://api.groq.com/openai/v1`

Do AI se posila pouze compact JSON summary:
- tickery
- vahy
- agregovane metriky
- deterministic score
- flagy
- kontext (risk profile, horizon)

Do AI se neposilaji raw historicka cenova data.

Pokud AI vrstva selze (chybi klic, timeout, API error), aplikace:
- nespadne
- vrati deterministic fallback komentare
- zachova vsechny ostatni vypocty a exporty

## 6) Jak funguje PDF/export pipeline

Export je v `src/reporting/export.py`:
- vice-strankovy PDF report (`BytesIO`) pres `matplotlib.backends.backend_pdf.PdfPages`
- obsahuje: shrnuti, vstupy, metriky, score+flagy, korelace, simulace, grafy, AI rozbor, doporuceni
- grafy jsou vkladane jako obrazky (matplotlib figure)
- robustni error handling: pri chybe exportu zustava app funkcni

Dostupne exporty v UI:
- `Export PDF`
- `Export data` (CSV)
- `Export full report` (JSON)

## 7) Struktura projektu

```
config/
src/
  ai/
    ai_review.py
  analytics/
    portfolio_metrics.py
    scoring.py
  reporting/
    export.py
  data/
  optimization/
  simulation/
  visualization/
ui/
  streamlit_app.py
.streamlit/
  secrets.toml.example
requirements.txt
```

## 8) Poznamky k dalsimu doladeni

- Pridat automatizovane testy (unit/integration) pro scoring, validace vstupu a exporty.
- Volitelne pridat fallback model switch v AI vrstve.
- Volitelne rozsirit data export o ZIP bundle (vice CSV souboru).

## 9) Modular dashboard vrstva (nove)

Aplikace nyni obsahuje modularni quant stack v `src/analytics/modular/`:
- pluggable model framework (bayesian/classical/ml registry)
- portfolio model template: Black-Litterman (`black_litterman`)
- pluggable signal framework
- additional signals: `black_litterman_tilt`, `sentiment_adjusted`
- summary engine nad modely + signaly
- news relevance vrstvu (provider je zamenitelny) + sentiment scoring
- deterministic no-look-ahead backtest vrstvu
- persistence run history (`data/run_history/*.json`) + compare

V UI pribyly taby:
- Data
- Models
- Signals
- Backtest
- News
- Summary
- History
- Compare

Kazdy run uklada:
- timestamp, config, universe, date range
- model/signal outputs
- metrics + summary + news relevance + sentiment aggregate

## 10) Testy

Spusteni testu:

```bash
python -m pytest -q
```

Aktualni testy pokryvaji:
- interface consistency
- no-look-ahead behavior
- news relevance scoring
- sentiment scoring
- black-litterman output consistency
- summary aggregation
- persistence round-trip
- failure isolation
- deterministic backtest example

## 11) Methodology & Validation (Wharton presentation readiness)

Wharton Cockpit now includes a **Methodology & Validation** module. It reports
an internal evidence-quality score rather than claiming a percentage of future
forecast accuracy. The module includes:

- 95% moving-block bootstrap intervals for annualized return, volatility,
  Sharpe ratio, daily VaR and CVaR;
- Monte Carlo convergence error, an analytic GBM cross-check and reproducible
  seeded runs;
- skewness, excess kurtosis, normality and lag-1 dependence diagnostics;
- a causal walk-forward baseline with turnover and transaction costs;
- explicit validation gates and limitations for a competition presentation.

The score is **not an official Wharton rating or endorsement**. QuantSim is
currently suitable as structured decision support, not as a validated
forecasting system. Claiming predictive accuracy requires a frozen strategy,
nested walk-forward testing of the full ensemble, untouched holdout data and
comparison with investable benchmarks after costs. See
`docs/MODEL_VALIDATION.md` for the methodology and interpretation rules.

## 12) Wharton analytical workflow

The Wharton Cockpit contains an analytical Strategy Lab rather than a report
generator. Its shared SQLite/Turso data model stores:

- a measurable Client Mandate with goal buckets, horizons, liquidity needs,
  risk tolerance, exclusions and required holding tags;
- append-only Strategy Rulebook versions with position, sector, cash,
  diversification, turnover, beta and approved-universe limits;
- a transparent 0-100 strategy-alignment diagnostic, client-goal and sector
  drift, HHI, effective holdings, cash weight and holding-level violations;
- per-holding thesis monitoring with bear/base/bull cases, catalysts, risks,
  invalidation conditions, conviction and scheduled review dates;
- a read-only WInS CSV/Excel reconciliation that flags missing or extra
  positions and quantity, cost-basis and market-value differences without
  overwriting either source;
- an analyst-controlled approved-security universe that remains explicitly
  unofficial until the current Wharton list is loaded;
- peer-relative company analytics across valuation, growth, profitability and
  balance-sheet risk, including coverage-adjusted percentiles;
- user-entered Porter Five Forces and SWOT analysis. Missing qualitative inputs
  are never inferred or invented by the application.

The 0-100 values are internal process diagnostics, not Wharton scores, credit
ratings, recommendations or return forecasts.
