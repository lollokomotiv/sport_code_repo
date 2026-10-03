# Promuovere un risultato da notebook a script

Un notebook di `explorations/` serve a capire: può essere sporco, girare su una
partita, avere celle eseguite fuori ordine. Un numero che esce dal workbench —
in un post, in un piano, nella submission — deve invece venire da un comando che
chiunque può rieseguire. La promozione è il passaggio fra le due cose.

Non è burocrazia: **la promozione è anche un controllo**. Un numero misurato su
una partita spesso cambia su venti (è la domanda del CLAUDE.md, §8: "regge se
cambio partita?"). Se cambia, lo si scopre prima di pubblicarlo.

## Quando serve

- il numero o la figura stanno solo in un notebook;
- il numero sta in un notebook e in un report, ma i due non coincidono;
- il report esiste ma non dice campione, copertura o limiti.

## I passi

1. **Isola.** Nel notebook, individua le celle che producono il numero. Scrivi
   in una lista: quali partite, quali filtri, quali scelte (parametri, soglie,
   esclusioni). Queste scelte vanno fissate **prima** di girare su tutte le
   partite e non si ritoccano dopo aver visto il risultato.

2. **Sposta le funzioni in `lib/`** se servono a più di un posto, e fai
   importare quelle al notebook. Il notebook resta la spiegazione; il codice
   vive in un posto solo.

3. **Scrivi lo script** `scripts/<nome>.py`:
   - gira su **tutte le partite** per default (`data.list_matches()`), o dichiara
     perché no;
   - accetta id di partita da riga di comando per lo sviluppo;
   - scrive `reports/<nome>.md` con la struttura qui sotto;
   - se il risultato andrà su X, ha una funzione `figura_post()` (o un flag
     `--post`) che genera i media con `lib/social.py` in `figures/post/`.

4. **Verifica l'equivalenza.** Sullo stesso sottoinsieme del notebook (stessa
   partita, stessi filtri) lo script deve dare lo stesso numero. Il modo più
   semplice è una cella nel notebook che chiama la funzione dello script e fa un
   `assert`, come §8 di `explorations/03`. Se i numeri non coincidono, uno dei
   due è sbagliato: ci si ferma e si capisce quale.

5. **Leggi il numero su tutte le partite** e confrontalo con quello del
   notebook. Se è cambiato molto, è un risultato in sé: va detto all'utente
   prima di qualunque post.

6. **Test, solo se c'è una proprietà con risposta nota** (un caso limite, una
   invarianza, un valore di riferimento). Mai un test sul risultato ("la
   correlazione supera 0,5"): è la regola dei `/goal` nel CLAUDE.md.

7. **Collega.** Il notebook rimanda al report; la riga in
   `explorations/README.md` dice che il risultato è stato promosso e dove.

## La struttura minima del report

```markdown
# <titolo: cosa misura>

Generato da `scripts/<nome>.py` (`python scripts/<nome>.py`), commit <hash>.

## Campione
partite, eventi o possessi, filtri applicati, esclusioni (con quanti esclusi)

## Copertura
quanta parte delle posizioni usate era osservata (`is_detected`), se il
risultato dipende da posizioni

## Risultati
i numeri, ciascuno con il suo confronto

## Limiti
cosa il risultato non dice; metriche di modelli SkillCorner usate; differenze
da un paper, se è una replica
```

`reports/tau_opp.md` è un esempio esistente: la sezione "Differenze dal paper"
fa da Limiti.

## Cosa non fare

- calcolare il numero del post nella shell "solo per questa volta";
- copiare un numero dal notebook nel report a mano;
- cambiare filtri o soglie dopo aver visto il risultato su tutte le partite;
- promuovere solo il numero che serve al post e lasciare nel notebook il
  controllo che lo ridimensiona.
