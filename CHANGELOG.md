# Changelog

## Necommitnuté zmeny oproti `71afbe9`

Porovnanie pracovného stromu s commitom `71afbe9` (`feature - Add KCM GTFS parser and E-VSP instance model`), vrátane nových nesledovaných súborov.

- **JSON schéma 1.0:** pridaná štrukturálna a sémantická validácia pri vytváraní, ukladaní a načítaní inštancií. Chybové hlásenia obsahujú cestu k problematickým údajom.
- **Oprava GTFS parsera:** zachovanie skutočných koncových zastávok pri chýbajúcich vzdialenostiach; kontrola časov, súradníc a kladných trvaní.
- **Nadväznosti:** `check_baseline()` používa rovnakú predvolenú prestávku ako `arcs()`. Diagnostika obsahuje všetky neplatné dvojice a ich časový deficit.
- **Energetický model:** pridaná metóda `dh_energy_kwh()` pre spotrebu prázdnych presunov.
- **Dátový kontrakt:** nabíjačky rozlišujú predpokladanú a potvrdenú infraštruktúru; `peak_scope="global"` stanovuje rozsah budúcej penalizácie špičky.
- **Inštancie:** všetky štyri existujúce JSON súbory migrované na novú schému.
- **Reprodukovateľnosť:** generátor zaznamenáva revíziu zdroja, SHA256 vstupov a parametre výberu.
- **Diagnostické nástroje:** pridaný `scripts/validate_instances.py` a výstup `instances/baseline_validation.json`.
- **Dokumentácia a prostredie:** pridané README, pripnuté závislosti a ignorovanie virtuálnych prostredí a Python cache.
- **Testovanie:** rozšírenie zo 6 na **45 úspešných testov**, vrátane chybných vstupov, energetických výpočtov a GTFS → JSON integrácie.
- **Existujúca lokálna zmena v `main.py`:** počet meraní upravený z 1024 na 1000.

### Známe obmedzenie

Inštancia `large` má pri päťminútovej prestávke 49 neplatných nadväzností; sú zdokumentované v diagnostickom výstupe. Odhady presunov neboli umelo upravené tak, aby pôvodné bloky prešli kontrolou.
