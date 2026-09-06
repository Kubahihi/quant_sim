# Vypořádání auditu QuantSim — 6. září 2026

Audit z 5. září byl porovnán s pracovním stromem nad commitem
`a734a30f48826b778c6c561588492ffcdbd7db5a`. Pracovní strom již obsahoval
rozpracované opravy. Ty byly přezkoumány, doplněny a ověřeny společně.
Změny jsou připravené lokálně; tento zásah není nasazením do produkce.

## Přehled 23 nálezů

| Nález | Vypořádání v aktuálním kódu |
| --- | --- |
| 01 — Klíč v historii | Konfigurace záznamu má seznam povolených analytických parametrů. Rekurzivní odstranění tajných polí chrání serializaci i čtení starších záznamů. Místní historie byla vyčištěna; vzdálené kopie vyžadují provozní kontrolu níže. |
| 02 — Historický CVaR | Sdílená definice Expected Shortfall započítává přesnou pravděpodobnostní hmotnost ocasu včetně zlomku hraničního scénáře. Používá ji analytika, optimalizační report, bootstrap a výstup simulace. |
| 03 — Časové zóny | Ceny i objemy se před spojením normalizují na místní datum burzovní seance. Duplicitní denní seance se odmítají. |
| 04 — Měny DCF | Příprava vstupů uchovává měnu výkazů, kotace a použitý kurz. Odlišné měny vyžadují explicitní kladný konečný převod; neznámá měna výsledek blokuje. Převádějí se i peněžní vstupy WACC. |
| 05 — Náhradní hotovost jako úspěch | Maximální Sharpe rozlišuje `optimal`, `fallback_feasible` a `failed`. Náhradní hotovost po selhání solveru má `success=False`, prázdné doporučené váhy a samostatné diagnostické `fallback_weights`. |
| 06 — Sazební konvence Sharpe | Riziková aktiva i hotovost mají anualizované aritmetické výnosy. Efektivní roční bezriziková sazba se před porovnáním převádí přes periodickou sazbu. Sjednoceny optimalizátory, frontier, Black–Litterman i reporty. |
| 07 — Likvidita odchozích pozic | Likvidační prodeje se kontrolují před optimalizací i v celém výsledném pokynu. Spotřebují společný limit obratu; nákupy zbývajících titulů vycházejí ze skutečných držených vah. Neproveditelná změna zachová původní držení a označí neúspěšné okno. |
| 08 — Mandát po zaokrouhlení | Konečné počty kusů a zbytková hotovost se znovu kontrolují proti vahám, sektorům, betě, hotovosti, způsobilosti titulů, obratu a počtu pozic. Při zadaném limitu volatility se kontroluje i riziko skutečných konečných vah. |
| 09 — Staré ceny | Doplnění poslední ceny je omezeno na 3 pozorované seance a 7 kalendářních dnů. Delší výpadek zastaví dotčený výpočet. Stejná kontrola chrání i benchmark v cockpit. |
| 10 — Neměnné NAV | Strategie i srovnávací portfolio mají vlastní průběžné NAV po výnosech a nákladech. Pozdější objemy pokynů a tržní dopad používají toto NAV. |
| 11 — Chybějící ADV | Chybějící data nemohou tiše vypnout požadovanou ochranu. Neověřitelný obchod se zablokuje; UI vyžaduje vědomé vypnutí exekučního modelu pro orientační analýzu. Dostupné hodnoty ADV se zachovávají. |
| 12 — Minimum pokynu | Minimální hodnota se znovu kontroluje po úpravě nákupu podle dostupných peněz a poplatků. |
| 13 — Úplná ztráta | Jednoduchý výnos −100 % je platný; hodnoty pod ním zůstávají neplatné. Kumulativní výsledek zůstane na −100 %. Simulace znovu nekoupí zaniklý titul; úplná ztráta účtu ukončí porovnání na daném pozorování a přizná případné zkrácení hodnocení. |
| 14 — Beta | Kovariance i variance trhu se počítají ze stejných společných pozorování. |
| 15 — Nečíselné DCF | Validace zahrnuje hotovost, dluh, cenu, akcie a konečnost výsledků, včetně relativního podhodnocení. Nečíselné ocenění nemá `available=True`. |
| 16 — Identita a backend historie | Quant Platform předává osobní `user_id` při zápisu i čtení. Wharton používá výslovně týmovou historii ve stejném databázovém spojení jako ostatní týmová data. Osobní a týmový rozsah se nesmějí kombinovat. |
| 17 — Prázdný renderer historie | Renderer přijímá slovníkové záznamy vracené úložištěm. Test ověřuje zachované ID, ticker a tabulku. |
| 18 — Týmová session | Platnost je nejvýše 8 hodin a nečinnost nejvýše 30 minut. Kontrola databázového účtu při dalším běhu UI odvolá session po změně hesla, role, identity nebo odstranění účtu. |
| 19 — Přehled API | Počet a úspěšnost uzavřených obchodů používají společné období posledních 30 kalendářních dnů v UTC včetně dneška. Invalidované obchody se nepočítají jako uzavřené. Události se řadí před omezením počtu. |
| 20 — Watchlist API | Poslední analytické univerzum se načítá přes stejnou historii jako osobní databázový zápis. Lokální výchozí cesta už nezávisí na pracovním adresáři procesu. Historie se řadí podle časové značky záznamu. |
| 21 — Test bez soukromých secrets | UI test používá prázdnou konfiguraci, vlastní dočasnou databázi a syntetické přihlášení. Platný DCF příklad obsahuje explicitní měny. Produkční konfigurace se do ověřovací kopie nekopíruje. |
| 22 — EWMA vydávaná za GARCH | Chybějící `arch` znamená nedostupný GARCH s vysvětlením. EWMA se vykazuje samostatně a nezíská v agregaci druhý hlas pod jiným názvem. |
| 23 — Runtime | Při této kontrole už místní prostředí používalo požadovaný Python 3.12.13. Kontrola runtime a parity manifestů prošla; verze nebyla kvůli testu uvolněna. |

## Konkrétní kontrolní příklady

- Jeden výnos −20 % a dvacet nulových výnosů dává ES95 přibližně
  **19,047619 %**, shodně v analytice a reportu optimalizátoru.
- Dvě stejné čtyřicetidenní trajektorie s časovými zónami Londýna a New Yorku
  zůstávají na 40 řádcích a mají korelaci 1.
- Série s nulovým průměrným periodickým nadvýnosem má Sharpe 0 v analytice
  i optimalizaci; test zahrnuje kladnou i zápornou bezrizikovou sazbu.
- Syntetické výkazy v TWD po zadaném převodu 1/25 dávají stejnou cenu jako
  výkazy přímo v USD. Bez převodu se ocenění nevydá jako dostupné.
- Omezení původních vah 40/35/25 % na dvě pozice nemůže úspěšně projít
  40% stropem. Samostatný příklad kontroluje překročení volatility.
- Po odchodu aktiva s vahou 50 % dokáže optimalizátor dodržet omezený nákup
  dalšího aktiva a přesunout zbytek do likvidnějšího titulu. Rozpočet obratu
  zahrnuje prodej i nákupy.

Definice diskrétního ES vychází z
[Rockafellar–Uryasev: Conditional Value-at-Risk for General Loss Distributions](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf).
Všechny uvedené číselné příklady používají syntetická data, nikoli aktuální
tržní doporučení.

## Vyčištění místní historie

Kontrola zahrnula **27 záznamů**: 14 souborů JSON a 13 databázových záznamů.
Byly prohlédnuty dvě místní databáze. V sedmi záznamech bylo neprázdné pole
`news_api_key`; platnost těchto hodnot u poskytovatele nebyla zjišťována.

Tajná pole, včetně prázdných historických polí, byla odstraněna ze všech
27 dotčených záznamů. Analytické výsledky, portfolia a přihlašovací údaje účtů
se tímto čištěním neměnily. Následná kontrola nenašla žádné zbývající tajné
pole v prověřených záznamech. Hodnoty klíčů nebyly součástí výstupu kontroly.

Opakovatelný nástroj `scripts/sanitize_run_history.py` standardně pouze kontroluje;
přepínač `--apply` odstraní tajná pole z místní historie. Nástroj má regresní test
pro kontrolní režim, vyčištění JSON i databáze, opakovaný běh a zachování jiných
databázových záznamů.

Vzdálená databáze, externí zálohy a případné staré diskové kopie nebyly tímto
místním čištěním prověřeny ani vyčištěny. Pokud neprázdné hodnoty patřily
skutečně používanému klíči, jeho vlastník má prověřit tyto kopie a klíč vyměnit
u poskytovatele. Aktivní konfigurace poskytovatele nebyla změněna.

## Ověření

**Závěrečný běh: 1 050 testů prošlo, 0 selhalo, 175,65 s.** Varování byla
nastavena jako chyby (`-W error`). Řádkové pokrytí `src` a `ui` je **60,62 %**
a překračuje projektový limit 57 %. Samotné procento pokrytí není zárukou
správnosti; kritické opravy mají konkrétní doménové regresní příklady.

Kontrola syntaxe, projektový lint, skutečný start Streamlit serveru,
kompatibilita 123 instalovaných balíčků a přesná verze Pythonu prošly.
Aktuální kontrola instalovaných závislostí pomocí `pip-audit --local --strict`
nenašla známé zranitelnosti. Tato kontrola není penetračním testem nasazení.

Strojové podklady jsou v místním adresáři `build/audit-2026-09-06/`:
[výsledky testů](../build/audit-2026-09-06/pytest-results.xml),
[protokol](../build/audit-2026-09-06/pytest.log),
[pokrytí](../build/audit-2026-09-06/coverage.json),
[kontrola závislostí](../build/audit-2026-09-06/dependencies.log)
a [souhrn ověření](../build/audit-2026-09-06/verification.json).
Regresní příklady jsou v `tests/test_audit_regressions.py`; širší kontrolu
zajišťují existující testy analytiky, optimalizace, API, přihlášení a UI.

Ověření běží v oddělené kopii bez soukromých portfolií, databází a `secrets.toml`.
Na tomto počítači potřebuje pytest vlastní dočasný adresář: dřívější sdílený
adresář pytestu má nedostatečná přístupová oprávnění. Do jeho oprávnění se
nezasahovalo. Při dokončování testů byl doplněn chybějící údaj o měnách do
syntetického příkladu ocenění, aby správně testoval dostupné DCF.

## Zbývající meze

Denní datum seance nesynchronizuje skutečné zavírací okamžiky různých burz.
Limit doplňování cen není plnohodnotný kalendář všech burz. DCF samo nevyhledává
kurz ani neověřuje poměr ADR a podkladových akcií; u takového instrumentu je
nutné ověřit i jednotku počtu akcií. Odhad nákladů a lotový plán nejsou
potvrzením proveditelnosti od brokera ani důkazem celočíselného optima.

Původně přiznaná omezení predikční validace, nekalibrované modelové confidence,
Monte Carlo, sdílených hesel a nehotového rizikového API zůstávají platná.
Opravy výpočtů samy neprokazují predikční úspěšnost strategie. Kompletní
kalibrace na oddělených datech je další výzkumný úkol.
