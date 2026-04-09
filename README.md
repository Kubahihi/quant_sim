# Quant Investment Platform

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
