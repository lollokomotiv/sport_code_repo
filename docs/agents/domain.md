# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This repo is multi-context: each top-level `*_project/` folder is an independent project with its own domain, data and rules.

## Before exploring, read these

- **`CONTEXT-MAP.md`** at the repo root: it points at one `CONTEXT.md` per project. Read the one for the project you're working in (and any other the topic touches).
- **`<project>/CONTEXT.md`**: the glossary for that project.
- **`<project>/docs/adr/`**: read ADRs that touch the area you're about to work in.
- **`docs/adr/`** at the root: only for decisions that span the whole repo.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

```
/
├── CONTEXT-MAP.md
├── docs/adr/                          ← repo-wide decisions
├── analytics_cup_project/
│   ├── CONTEXT.md
│   └── docs/adr/                      ← project-specific decisions
├── xgoals_project/
│   ├── CONTEXT.md
│   └── docs/adr/
├── totosport_project/
├── tennis_project/
├── snooker_project/
└── table_tennis_project/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in the project's `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids. A term defined in one project's glossary doesn't carry over to another project.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_
