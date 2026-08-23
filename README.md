# Quant Investment Platform

![Lines of Code](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.codetabs.com%2Fv1%2Floc%3Fgithub%3DKubahihi%2Fquant_sim&query=%240.linesOfCode&label=Lines%20of%20Code&color=blue)

Streamlit aplikace pro vyhodnoceni investicniho portfolia:
- validace tickeru a vah
- nacitani trznich dat z Yahoo Finance
- metriky, score, flagy a rizikovy rozbor
- volitelny doplnkovy komentar pres Groq API
- export vsech vysledku do vice-strankoveho PDF + CSV + JSON
- portfolio tracker pro dluhopisova ETF i jednotlive dluhopisy vcetne kuponu, FX, YTM, durace, DV01 a cash-flow kalendare
- jeden Research Workspace pro screener, company analysis, fixed income a real assets; investicni teze se uklada jen do canonical Security Dossier
- jeden Portfolio -> Risk & Scenarios workspace pro benchmark, faktory, FX, stresy, simulace a pokrocile diagnostiky
- behavioralni profil klienta v Client & Policy -> Mandate & Strategy: transparentni dotaznik, reakce na drawdown, kontrola souladu s deklarovanou toleranci a konkretni rozhodovaci guardraily
- Client Goal Outlook: pravdepodobnost splneni penezniho cile, expected/median/P10 terminal wealth a conditional shortfall pro current/optimalizovana portfolia pod stejnymi bootstrap scenari
- vypocitany Brinson-Fachler attribution v Report & Pitch z reconciled sektorovych vah a vynosu; allocation, selection a interaction se nezadavaji volnym textem
- Wharton bond case s kontrolou zpusobilosti ve WInS, vazbou na cil klienta, position-sizing limitem, pitch-defense otazkami, relative-value shortlistem a exportem pracovniho investicniho memo
- rucni jednotlive dluhopisy primo v Quant Enginu: smluvni parametry, YTW/durace/DV01 a transparentni ETF proxy pro kovarianci, optimalizaci, Monte Carlo a stresove scenare
- liquidity-aware portfolio construction s 30dennim dollar-ADV, spreadem, square-root market impactem a limitem ucasti na dennim objemu
- executable trade plan s celymi loty, minimalni hodnotou obchodu, cash kontrolou, poctem pozic a tax-aware vyberem lotu
- volitelna point-in-time rolling validace s lagovanou historii clenu univerza a fail-closed kontrolou chybejicich delisting returnu
- produkcni Runtime & build panel s identifikaci commitu a serverovym medianem/p95 poslednich rerunu

Konvence oceneni a prace s dluhopisovymi daty jsou popsane v [docs/FIXED_INCOME.md](docs/FIXED_INCOME.md).
Metodika komoditnich proxy, stresu pozice a jejich omezeni je popsana v [docs/COMMODITIES.md](docs/COMMODITIES.md).
Metodika menovych expozic, rizikovych metrik, stresu a optimalizace hedge je popsana v [docs/CURRENCY_RISK.md](docs/CURRENCY_RISK.md).

Metodika konstrukce portfolia, sdilenych odhadu, omezeni a rolling
out-of-sample reoptimalizace je popsana v
[docs/PORTFOLIO_OPTIMIZATION.md](docs/PORTFOLIO_OPTIMIZATION.md).
Metodika behaviorálního profilu, skore, evidence a governance omezeni je popsana v [docs/BEHAVIORAL_PROFILE.md](docs/BEHAVIORAL_PROFILE.md).
Metodika goal-funding bootstrapu, cash flow a interpretacnich omezeni je popsana v [docs/GOAL_FUNDING.md](docs/GOAL_FUNDING.md).

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

Pro reprodukovatelnou produkcni nebo CI instalaci pouzijte uzamcene verze:

```bash
uv venv --python 3.12.13
uv pip sync requirements.txt
```

`requirements.txt` je primo produkcni lock instalovany Streamlit Cloudem.
Prime zavislosti se meni v `requirements.in`; vyvojove a testovaci prostredi
se sestavi pres `requirements-dev.lock`. Soubor `.dependency-cutoff` zmrazi
casovy pohled na balickovy index, aby regenerace locku zustala opakovatelna.

### Volitelne nastaveni GROQ_API_KEY

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

Explicitne oznacte lokalni prostredi:

```powershell
$env:QUANT_SIM_ENV="development"
```

```bash
export QUANT_SIM_ENV=development
```

```bash
streamlit run ui/streamlit_app.py
```

## 2) Deploy na Streamlit Cloud

1. Nahrajte repozitar do GitHubu.
2. Ve Streamlit Cloud zvolte repo a soubor `ui/streamlit_app.py`. V Advanced
   settings explicitne zvolte Python 3.12, tedy stejnou minor verzi jako v CI.
3. Do sekce **Secrets** vlozte hodnoty podle `.streamlit/secrets.toml.example`. Pro produkci jsou povinne:

```toml
QUANT_SIM_ENV = "production"
WHARTON_SHARED_PASSWORD = "replace-with-the-shared-password"

TURSO_DATABASE_URL = "libsql://your-database.turso.io"
TURSO_AUTH_TOKEN = "your-turso-auth-token"
TURSO_SYNC_INTERVAL_SECONDS = 30

[wharton_users]
Jakub = "strong-unique-password"
"Lukáš" = "strong-unique-password"
Martin = "strong-unique-password"
"Matěj" = "strong-unique-password"

[storage]
STORAGE_BACKEND = "r2"
R2_BUCKET = "your-r2-bucket"
R2_ENDPOINT_URL = "https://your-account-id.r2.cloudflarestorage.com"
R2_ACCESS_KEY_ID = "your-r2-access-key-id"
R2_SECRET_ACCESS_KEY = "your-r2-secret-access-key"
```

V produkci `WHARTON_SHARED_PASSWORD` nastavi jedno spolecne heslo pro vsechny ctyri
uctu a ma prednost pred `[wharton_users]`. Kvuli kompatibilite online nasazeni
akceptuje jako sdilene heslo take drive pouzivany top-level `WHARTON_PASSWORD`.
Lokalni development pouziva
jednotliva hesla z `[wharton_users]` a chybejici ucty muze doplnit hodnotou
`WHARTON_PASSWORD`. Heslo musi mit 10–72 UTF-8 bajtu. Po zmene prihlasovacich
udaju aplikaci restartujte, aby se aktualizovaly ulozene otisky hesel.

4. Nastavte hlavni soubor na `ui/streamlit_app.py` a nasadte aplikaci. Po prvnim spusteni se vytvori sdilena databazova struktura a aktualizuji se role zakladnich uzivatelu.

Vsechny realne hodnoty patri pouze do Streamlit Cloud Secrets. Bez Turso by se prihlaseni a sdilena data po redeployi neuchovala; bez R2 by se neuchovaly nahrane soubory.

`QUANT_SIM_ENV` muze byt systemova promenna nebo top-level Streamlit secret.
Systemova promenna ma prednost. Streamlit server bez explicitni hodnoty se v
perzistentnich vrstvach posuzuje fail-closed jako produkce.

Poznamka: Pokud klic chybi nebo Groq neodpovi, aplikace bezi dal a pouzije zakladni komentar podle pravidel.

### Sdilena online databaze

Decision log, historie jeho uprav, data dashboardu a verzovane makro snapshoty
pro regionalni analyzu se ukladaji do sdilene Turso/libSQL databaze, jakmile
jsou nastaveny `TURSO_DATABASE_URL` a `TURSO_AUTH_TOKEN`. Aplikace pri startu
automaticky vytvori nebo zaktualizuje tabulky a po kazdem ulozeni synchronizuje
zmeny online. Zmeny z jinych bezicich replik stahuje nejvyse jednou za interval
`TURSO_SYNC_INTERVAL_SECONDS` (vychozi hodnota je 30 sekund), aby kazdy Streamlit
rerun necekal na sit. Vlastni uspesne zapisy jsou v dane instanci viditelne
okamzite. Makro snapshoty pro referencni rok 2024 maji sestihodinovou
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
- fallback text pouzitelny i bez externi sluzby

## 5) Jak funguje volitelny komentar

Do volitelne sluzby se posila pouze compact JSON summary:
- tickery
- vahy
- agregovane metriky
- deterministic score
- flagy
- kontext (risk profile, horizon)

Do sluzby se neposilaji raw historicka cenova data.

Pokud volitelna sluzba selze (chybi klic, timeout, API error), aplikace:
- nespadne
- vrati deterministic fallback komentare
- zachova vsechny ostatni vypocty a exporty

## 6) Jak funguje PDF/export pipeline

Export je v `src/reporting/export.py`:
- vice-strankovy PDF report (`BytesIO`) pres `matplotlib.backends.backend_pdf.PdfPages`
- obsahuje: shrnuti, vstupy, metriky, score+flagy, korelace, simulace, grafy a doporuceni
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
  streamlit_app.py      # lightweight launcher
  quant_platform.py     # analytical workspace loaded on demand
.streamlit/
  secrets.toml.example
requirements.txt
```

## 8) Release a provozni pripravenost

- CI pri kazde zmene kontroluje lockfile, kompilaci, correctness lint, cely test suite,
  warning-free beh a minimalne 57% celkove coverage.
- Pred release se kontroluje cela Git historie na unikla tajemstvi a presne
  instalovane zavislosti na zname bezpecnostni zranitelnosti.
- Release artefakt obsahuje overeny CycloneDX SBOM a uplny licencni inventar
  vsech produkcnich balicku.
- Uspesny release vytvari 90denni artefakt svazany s plnym commit SHA a SHA-256
  hashi vsech nasazovanych souboru. GitHub attestation potvrzuje jeho puvod a
  stejny manifest overuje rollback kandidata.
- Produkcni rezim odmita lokalni SQLite misto Turso, lokalni file storage misto R2
  a nebezpecne API nastaveni jako wildcard CORS nebo vypnutou autentizaci.
- Detailni release postup je v [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md).

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
- a causal walk-forward baseline plus rolling portfolio re-optimization with
  turnover costs, an equal-weight comparator, and optional lagged point-in-time
  universe control;
- explicit validation gates and limitations for a competition presentation.

The score is **not an official Wharton rating or endorsement**. QuantSim is
currently suitable as structured decision support, not as a validated
forecasting system. Claiming predictive accuracy requires a frozen strategy,
nested walk-forward testing of the full ensemble, untouched holdout data and
comparison with investable benchmarks after costs. See
`docs/MODEL_VALIDATION.md` for the methodology and interpretation rules.

## 12) Wharton competition workflow

The default workspace follows one competition process instead of exposing every
model as a separate destination:

1. **Home** — next action, owner, blocker and competition readiness.
2. **Client & Policy** — measurable goals, behavior, rulebook and alignment.
3. **Research** — screening and analysis followed by one canonical Security
   Dossier. The dossier contains evidence, valuation, KPIs, catalysts and exit
   discipline; it never contains an investment vote.
4. **Decisions** — the Investment Committee is the sole investment-voting
   system. Blind initial view, discussion, final vote, authorization and sizing
   are stages of the same lifecycle.
5. **Portfolio** — WInS import, reconciliation, portfolio outcomes, client-goal
   probability and one consolidated risk/scenario workspace.
6. **Deliverables** — report, pitch rehearsal, rules and submission evidence.

The shared SQLite/Turso model has explicit authorities:

- Client Mandate stores investable capital and goal-level target amount,
  horizon, capital allocation, contributions/withdrawals, inflation and
  nominal/real basis;
- canonical Security Dossiers replace the legacy Thesis Monitor, Catalyst
  Calendar and security-level review inputs;
- the active Authoritative Universe replaces the parallel analyst-approved
  universe as the eligibility gate;
- the canonical Investment Committee lifecycle replaces the legacy Decision
  Journal for all new decisions;
- the signed WInS reconciliation is the only portfolio snapshot allowed into
  reporting;
- Research Evidence Registry is the source authority; dossiers store registry
  IDs and reports freeze immutable snapshots of verified registry records;
- report performance attribution is calculated with Brinson-Fachler from
  reconciled sector inputs and must reconcile to portfolio return.

Legacy thesis, universe, decision and position tables remain readable during
the transition. A canonical-first, read-only adapter uses a legacy row only
when no canonical record exists for that ticker; legacy editors are not part of
the primary competition navigation.

The Client Goal Outlook uses a seeded historical row bootstrap and applies the
same sampled shocks to every candidate portfolio. It reports planning ranges,
not a return forecast. A monetary target and cash-flow schedule do not remove
model or parameter uncertainty.

The 0-100 values are internal process diagnostics, not Wharton scores, credit
ratings, recommendations or return forecasts.
