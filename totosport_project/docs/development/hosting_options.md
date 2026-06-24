# Opzioni di hosting — TotoSport

Confronto delle possibilità per pubblicare TotoSport (Fase 11). Contesto: gioco
**privato** per ~20 amici, uso prevalente da **mobile**, serve **HTTPS** (anche
per rendere la PWA installabile). Non è un prodotto SaaS: la priorità è
**semplicità + costo basso**, non la scalabilità.

> ⚠️ I prezzi e i tier gratuiti cambiano spesso: verifica sempre il listino
> attuale del provider prima di scegliere.

---

## Cosa va hostato

L'app ha tre componenti:

| Componente | Cos'è | Come si serve |
|---|---|---|
| **Frontend** | React SPA → file statici dopo `npm run build` | static hosting o Nginx |
| **Backend** | FastAPI (uvicorn) | processo sempre attivo |
| **Database** | PostgreSQL 16 | servizio gestito o container |

Il frontend in produzione viene buildato con `VITE_API_BASE_URL=/api` e le
chiamate `/api` vengono inoltrate al backend (reverse proxy).

---

## Opzione A — Piattaforma managed (Render / Railway)

Deploy automatico dal repo GitHub: build, HTTPS e Postgres gestiti dal provider.

- **Sforzo:** minimo — colleghi il repo, ogni `git push` ridistribuisce.
- **Costo (indicativo):**
  - Tier **gratuito** usabile per i test, ma con limiti: il backend "si addormenta"
    dopo inattività → primo accesso lento (~30–60s); il Postgres gratuito ha
    scadenza/limiti.
  - **Sempre attivo:** ~**7–15 $/mese** (backend + Postgres; il frontend statico
    è di norma gratuito).
- **Pro:** zero gestione server, HTTPS automatico, rollback facili, deploy al push.
- **Contro:** cold-start sul tier gratuito; il costo always-on cresce un po';
  un filo di lock-in sulla piattaforma.
- **Per chi:** "non voglio toccare un server".

Varianti equivalenti: **Railway** (DX semplicissima, Postgres one-click, billing
a consumo), **Fly.io** (gira container Docker, vicino agli utenti — più config).

---

## Opzione B — VPS economico + Docker (Hetzner / DigitalOcean)

Una piccola macchina dove gira il nostro `docker-compose.prod.yml` + Nginx +
HTTPS (Certbot/Let's Encrypt). **È lo scenario per cui sono già scritti i docs
della Fase 11** (`11_deployment.md`).

- **Sforzo:** medio — setup iniziale una volta, poi aggiornamenti con `deploy.sh`.
  Config e piano di backup già previsti nei docs.
- **Costo (indicativo):** ~**4–5 €/mese** fisso e prevedibile, tutto in una macchina.
- **Pro:** controllo totale, costo minimo prevedibile, niente lock-in, un solo
  posto per frontend + backend + DB.
- **Contro:** manutenzione (aggiornamenti SO, backup, sicurezza, rinnovo
  certificati) a carico tuo — leggera, ma esiste.
- **Per chi:** vuoi pagare poco e avere controllo, e non ti spaventa un minimo di
  setup iniziale (che abbiamo già documentato).

Provider tipici: **Hetzner** (CX/CAX, ~3,8–5 €/mese), **DigitalOcean** (~6 $/mese),
**Contabo** (economici, risorse abbondanti).

---

## Opzione C — Oracle Cloud "Always Free"

Una VM **sempre gratuita** (tipicamente ARM) che gestisci come un VPS.

- **Sforzo:** medio (come il VPS) + iscrizione un po' macchinosa.
- **Costo:** **0 €/mese**, davvero gratis e sempre attiva.
- **Pro:** gratis per sempre, risorse abbondanti (CPU/RAM generose nel tier free).
- **Contro:** signup con carta di credito; a volte difficile trovare disponibilità
  delle istanze ARM gratuite; self-managed come un VPS.
- **Per chi:** "gratis assoluto" è la priorità e non temi lo smanettamento.

---

## Variante trasversale: frontend separato dal backend

Il frontend (statico) può andare **gratis** su **Vercel / Netlify / Cloudflare
Pages** (deploy al push, HTTPS incluso), mentre backend + DB stanno su una delle
opzioni sopra. Riduce il carico sul server e dà al frontend una CDN globale.
Costo aggiuntivo: 0 €. Svantaggio: due posti da configurare invece di uno, e va
gestita la `VITE_API_BASE_URL` + il CORS verso il dominio del backend.

---

## Riepilogo

| Opzione | Sforzo | Costo/mese | Gestione server | HTTPS |
|---|---|---|---|---|
| **A — Managed (Render/Railway)** | basso | ~7–15 $ (free per test) | no | automatico |
| **B — VPS + Docker (Hetzner)** | medio | ~4–5 € | sì (leggera) | Certbot |
| **C — Oracle Always Free** | medio | 0 € | sì | Certbot |

### Raccomandazione
- Vuoi **zero gestione** e ti sta bene ~5–15 $/mese → **A**.
- Vuoi **costo minimo prevedibile e controllo**, con un po' di setup già
  documentato → **B** (la strada "ufficiale" dei nostri docs).
- **Gratis assoluto** è la priorità e non temi il self-managing → **C**.

In tutti i casi: dominio (~10 €/anno, opzionale ma consigliato per un URL
stabile e HTTPS pulito) e backup periodico del DB.
