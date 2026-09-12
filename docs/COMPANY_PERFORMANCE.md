# Zrychlení analýzy firem — 9. září 2026

Analýza firmy načítá nezávislé zdroje souběžně a detail zpracovává pouze
otevřenou záložku. V kontrolovaném místním měření načítání dat trvalo
**3,63× méně času** a opakované vykreslení přehledu mělo **o 41 % kratší dobu**.

## Měření

Porovnání s commitem `e579ac60dab37654dc66b4f588ff582a086c8227` na stejném
počítači, Python 3.12.13, Streamlit 1.61.1, pět opakování každé varianty.
Časy v tabulce jsou mediány.

| Kontrola | Původní verze | Upravená verze |
| --- | ---: | ---: |
| Načtení celého firemního snapshotu | 557,2 ms | 153,7 ms |
| Opakované serverové vykreslení přehledu firmy | 201,3 ms | 119,7 ms |
| Prvky vykreslené v testovaném přehledu | 160 | 41 |
| Volání skrytých analýz během pěti vykreslení | 20 | 0 |

Jde o **offline benchmark**, nikoli měření skutečné odezvy Yahoo, SEC, AI služby
nebo veřejně nasazené aplikace. Každý náhradní datový endpoint čeká 50 ms.
Při měření rozhraní jsou externí obohacení nahrazena okamžitou testovací odpovědí;
měří se serverové vykreslení po zahřátí importů, nikoli vykreslování v prohlížeči.
Výsledek ukazuje úsporu postupného čekání a zbytečné práce, ne garantovanou
rychlost při libovolném stavu sítě.

## Co se změnilo

- Data firmy se načítají nejvýše ve čtyřech souběžných úlohách. Každá úloha má
  vlastní objekt poskytovatele. Roční a čtvrtletní výkazy stejného typu zůstávají
  spolu, analýza geografických tržeb čeká na seznam příslušných výkazů.
- Dílčí výpadky zdrojů nadále vracejí dostupná data a konkrétní diagnostiku.
  Pořadí diagnostiky je stabilní bez ohledu na pořadí dokončení úloh.
- Devět záložek detailu firmy i roční/čtvrtletní výkazy používají stavové
  záložky. Skrytá záložka nespouští AI návrh DCF, hledání konkurentů, načítání
  evidence ani tvorbu tabulek. Použitý mechanismus je podporovaný připnutou
  verzí Streamlitu; viz [dokumentace dynamických kontejnerů](https://docs.streamlit.io/develop/concepts/design/layouts-and-containers).
- Hodnoty DCF předané aplikaci, výběr konkurentů a volba záložky se uchovávají
  při přepínání. Formuláře nadále vyžadují odeslání změn svým potvrzovacím tlačítkem.

## Ověření

- **1 052 testů prošlo**, včetně režimu, který považuje Python warnings za chyby.
- Pokrytí celé testovací sady **60,65 %**, nad projektovým limitem 57 %.
- Integrační kontrola všech devíti záložek a obou variant finančních výkazů;
  kontrola zachování upraveného DCF a ručně vybraného konkurenta.
- Souběžnost ověřená synchronizační bariérou: přesně čtyři souběžné požadavky,
  žádné sdílení objektu poskytovatele mezi pracovními vlákny, úplné výsledky
  a správný částečný výsledek při výpadku endpointu.
- Kontrola kódu, připnutých závislostí a start skutečného testovacího serveru prošly.

Benchmark je opakovatelný bez skutečných síťových požadavků:

```powershell
.venv\Scripts\python.exe scripts/benchmark_company_performance.py --output build/company-performance.json
```

Volba `--project-root` umožňuje stejným benchmarkem měřit jiný checkout.
Naměřená data se ukládají do JSON včetně jednotlivých opakování a verzí prostředí.
