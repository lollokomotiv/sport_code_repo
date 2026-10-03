[[tau_opp]]
Il concetto di tau_opp è legato alla velocità con cui il giocatore arriva sulla palla. 

In letteratura i riferimenti al pressing difensivo sono nei PDF: 
1. [[exPressV2_Contextual_Evaluation_Pressing.pdf]]
2. [[Quantifying_defensive_pressure_on_the_ball_carrier_in_soccer_based_on_minimum_arrival_time.pdf]] 
3. [[defensivepressurearticle_researchgate.pdf]]

PROBLEMA 1: la velocità in tau_opp diventa un fattore molto rilevante per il calcolo della pressione e, nei casi in cui due giocatori sono quasi contemporaneamente sul portatore, è difficile calcolare il portatore più vicino. 

[[01-scegliere-la-domanda]] --> considerazioni su Goal A e B

Problema 2: esistono due metriche molto simili messa a disposizione da SkillCorner che si chiama 
--> **time_to_impact** <-- 
e una di unravel_sports che si chiama 
--> **pressing_intensity** <-- 

**Tau_opp - time_to_impact - pressing_intensity**

Importante --> capire se ha senso portare avanti l'esplorazione dato che queste tre metriche hanno significati simili, e time_to_impact in particolare è molto simile a tau_opp che stavo utilizzando per portare avanti una tesi. 

DA TESTARE pressing_intensity di unravelsports

- [Notebook 03 — calcolo di τ_opp, §10 confronto con time_to_impact](../../explorations/03-calcolo-tau-opp.ipynb)
- [Notebook 04 — Pressing Intensity di unravelsports contro τ_opp](../../explorations/04-pressing-intensity-unravel.ipynb)
