# Laura Gao: posouzení mezer a aktualizace Quant Simu

Podklady: `Laura_Gao_assessment_CS.pdf` (stav k 17. 9. 2026, 13 stran) a
`Laura_Gao_Cashflow.xlsx` (Přehled, Vstupy, Akumulace, Rezerva).
Identifikátory SHA-256 podkladů jsou uložené u definice případu a v exportech.
Assessment je dodaný analytický podklad, nikoli nově ověřený originál soutěžních
pravidel. Jeho doporučené metriky a ilustrace jsou oddělené od zadaných toků.

## Co chybělo a co se změnilo

| Mezera původní aplikace | Doplněná funkce | Vazba na assessment |
| --- | --- | --- |
| Hlášky, že klientské zadání a výstupy ještě nejsou zveřejněné | Laura Gao Plan, zdroje, pevné vstupy, aktualizovaný checklist | s. 1–3, 9–11 |
| Obecný konečný cíl a stejný každoroční tok | Dva datované vklady, šest akumulačních období, deset plateb před výnosem | K01–K05 |
| Koncový majetek jako hlavní důkaz úspěchu | Společné splnění všech plateb; rok prvního nedostatku a neuhrazené částky | K08 |
| Bez oddělené rezervy a rozhodnutí o budově | Ocenění rezervy včetně první platby, přirážka, flexibilita, příspěvek z přebytku | K06–K10 |
| Bez analýzy rozhodnutí v roce 2031 | Samostatný podmíněný výpočet z výslovně zadaného stavu k 1. 1. 2031 | K11 |
| Riziko záměny 95. percentilu za pokrytí intervalu | 5.–95. percentil, nominální 90% pokrytí, empirické pokrytí, riziko pod dolní mezí | K12 |
| Bez společného modelu cíle a čerpání | Párované scénáře, čtyři etapy alokace, náklady, stresy a porovnání portfolií | K07, K13, K15 |
| Bez návazného textu pro partnery a reprodukovatelných dat | Podmíněný pracovní fundraisingový text, JSON s historií, parametry, scénáři a náhodnými indexy | K14–K15 |
| Riziko přenosu zisku WInS do klientské projekce | Projekce nemá vstup pro hodnotu ani zisk WInS; začíná pouze dvěma vklady | K16 |
| Výstupy pro starší obecný harmonogram | Termíny 9. 10., 23. 10., 6. 11. a 4. 12.; samostatné kontroly 3×100, 50 a 500 slov | s. 10–11 |
| Chybějící vazba klientského výpočtu na report | Uložení plánu do mandátu; připojení neměnné kopie výpočtu jako modelového důkazu | K17 |

Dosavadní optimalizace, dluhopisové analýzy, měnové riziko, WInS reconciliation,
Security Dossiers a rozhodování komise zůstávají stavebními částmi strategie.
Nový plán je propojuje s konkrétními klientskými platbami. Obecný Client Goal
Outlook zůstává samostatným nástrojem pro konečné cíle a výslovně upozorňuje,
že neprokazuje financování Lauřiných deseti plateb.

## Použití

1. **Client & Policy → Laura Gao Plan → Case & Cash Flows:** zadání, kalendář
   a všech 17 požadavků. Částky zadání zde nejsou volně měnitelnými odhady.
2. **Funding Model:** okamžité porovnání čtyř deterministických ilustrací
   z Excelu, včetně úplné tabulky plateb. Nejsou jim přiděleny pravděpodobnosti.
3. Pro pravděpodobnostní model spusťte **Quant Engine** s instrumenty růstové
   části i rezervy. Potřebné jsou alespoň dva téměř úplné společné kalendářní
   roky dat; pod pěti roky aplikace upozorní na malý vzorek. Výnosy musejí být
   nominální total returns v USD; uživatel tento předpoklad potvrzuje.
4. Zadejte alokace pro 2027–2030, 2031–2032, 2033–2036 a 2037–2041.
   Každá musí dávat 100 %. Hotovost či chybějící instrument se nedoplňuje
   implicitně. Doplňte diskont rezervy, přirážku, náklady, pravidlo příspěvku,
   flexibilitu, definici jistoty a zdůvodnění.
5. Pro komunikaci partnerům výslovně určete modelový stav na začátku 2031.
   Výchozí hodnota je označená jako ilustrace při 6% růstu, nikoli pozorovaný
   majetek či predikce. Samostatná projekce má dva roky do začátku 2033.
6. **Save assumptions and calculate** uloží předpoklady a souhrn do sdíleného
   mandátu. Změna jiných částí mandátu tento plán ani pracovní odevzdání nemaže.
   Výsledky označují naposledy spočítané vstupy; úprava formuláře vyžaduje nový
   výpočet. Detailní scénáře lze stáhnout jako reprodukovatelný JSON.
7. **Submissions**, dostupné také v **Deliverables → Report & Pitch →
   Submission Drafts**, uchovává původní poznámky přesně a kontroluje
   jednotlivé slovní limity. Odeslání do SurveyMonkey Apply se neprovádí.
8. **Report Evidence Studio → Claims & evidence** umožňuje po kontrole připojit
   uložený klientský výpočet. Má vlastní hash, zdrojový typ `model_output`
   a kopii vstupů i výsledků. Lze ho citovat v tvrzeních reportu; pozdější změny
   plánu nepřepisují zmrazené důkazy.

## Metodika a kontroly

Základ je nominální USD. Rok modelu = kalendářní rok − 2026.
První vklad má do začátku 2033 šest výnosových období, druhý pět:

`P2033 = 300000 × (1+g)^6 + 150000 × (1+g)^5`.

Ilustrační rezerva je anuita splatná předem:

`R = (1 + přirážka) × sum(50000 / (1+y)^k, k=0..9)`.

Skutečně vyčleněná rezerva = `min(P, R)`; nedostatek = `max(R−P, 0)`.
Přebytek = `max(P−R, 0)`. Příspěvek na budovu =
`podíl × max(přebytek − minimální flexibilita, 0)`.
Zbytek přebytku tvoří flexibilitu. Ta se v této implementaci dále neinvestuje
a nezachraňuje rezervu. Rezerva, příspěvek a flexibilita přesně rozdělují P.
Chybějící požadovaná flexibilita se vykazuje zvlášť.

Každá platba je omezena skutečně dostupným zůstatkem. Dluh, externí příjem ani
záporná hotovost se nepřipouštějí. Roční výnos a dodatečné roční náklady se
uplatní až na zůstatek po platbě. Po poslední platbě na začátku 2042 se další
výnos nepočítá. Diskont rezervy není její realizovaný výnos.

Bootstrap vybírá celé řádky historických ročních výnosů, společné všem aktivům.
Zachovává jejich vzájemné vztahy uvnitř roku, nikoli časovou závislost mezi
roky. Stejný seed a řádky platí pro porovnávaná portfolia. Kandidáti z Quant
Enginu drží během akumulace konstantní váhy a sdílejí etapovou rezervní politiku;
vlastní klientský plán může měnit váhy již před 2033. Optimalizace na stejné
historii je označena jako průzkumná, nikoli výsledek nezávislé validace.

Model ukazuje odděleně dostupnost rezervy v roce 2033, společné splnění deseti
plateb, splnění podmíněné vytvořením rezervy a splnění celého plánu včetně
flexibility. 95% Wilsonův interval měří pouze chybu konečného počtu simulací.
Výběr týmového prahu 95 % není požadavkem Whartonu.

Citlivosti: propad v 2032, o 2 procentní body nižší výnosy, o 1 bod vyšší
náklady, nejhorší rezervní roky na začátku, souběh ztrát aktiv a nižší diskont
rezervy. Nejde o scénáře s předepsanou pravděpodobností.

Interval pro partnery zahrnuje i nulové příspěvky a podfinancované scénáře.
Neúspěšné scénáře se nezahazují. Pokrytí celého intervalu se odlišuje od
jednostranné šance dosažení minima a od společného splnění provozu. Shody
na nule mohou zvýšit empirické pokrytí nad nominálních 90 %.

## Co stále vyžaduje tým nebo další podklady

- Skutečná strategie, vhodné a dostupné instrumenty rezervy, zvolený práh
  jistoty, limity rizika a zdůvodněná flexibilita. Výchozí ilustrace 6 %, 9 %,
  diskont 3 % a podíl 80 % nejsou doporučením ani klientským omezením.
- Aktuální ověřená pravidla obchodování a způsobilost nástrojů ve WInS.
  Veřejné [FAQ Whartonu](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/faq/)
  a [pravidla](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/rules-roles/)
  odkazují registrované týmy na SurveyMonkey Apply. Původní veřejná adresa
  `competition-deliverables` při kontrole vracela 404; detailní termíny proto
  zachovávají původ v dodaném assessmentu.
- Konečné instrukce Final Reportu a vzor školní dokumentace, očekávané po
  9. listopadu. Obecné stránkové rozpočty reportového studia jsou pracovní šablony.
- Pravdivé doklady tří provedených obchodů, vlastní reflexe a soulad s IPS.
  Kontrola textu nenahrazuje ověření WInS, formátu finálního PDF ani týmový podpis.
- Flat-rate rezerva není skutečně nakoupený žebřík dluhopisů. Samostatná
  dluhopisová analýza musí doložit ceny, splatnosti, likviditu, úvěrové riziko
  a měnovou expozici. ETF nezaručuje částku v konkrétním budoucím datu.

## Ověření implementace

Regresní testy porovnávají růstový scénář s uloženými hodnotami Excelu:
P = 733 923,626487 USD, R = 439 305,446094 USD, příspěvek =
235 694,544315 USD, flexibilita = 58 923,636079 USD.
Při nulových výnosech správně chybí 50 000 USD, první deficit nastane v 2042.
Další testy pokrývají načasování, dvojí odčítání, náklady, nedostatek rezervy,
flexibilitu, podmíněný interval 2031, párování scénářů, ukládání, neměnné
důkazy reportu a skutečné spuštění formulářů prostřednictvím Streamlit AppTest.
