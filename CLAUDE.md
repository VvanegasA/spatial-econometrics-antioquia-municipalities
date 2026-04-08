# AI / collaborator instructions — spatial econometrics (Antioquia municipalities)

This file orients assistants (Claude, Cursor, etc.) when working in this repository. **Read it before making structural or methodological changes.**

---

## Communication

- **With the repository owner:** use **Spanish** — clear, professional, and pedagogical when they ask for explanations.
- **Code:** identifiers, file names, docstrings (if any), **commit messages**, and inline comments — **English**.
- **Comment style:** use a natural, direct, and concise voice. Focus on *why* the code does something rather than over-explaining *what* it does. Avoid overly robotic or "AI-like" literal language.
- **Logging and outputs:** do not use exploratory emojis (✅, ❌, etc.). Use standard professional CLI tags (`[INFO]`, `[SUCCESS]`, `[WARNING]`, `[ERROR]`).
- **Learning goal:** the owner is building econometrics expertise. Explain **methodology, assumptions, and interpretation** in Spanish; keep the codebase readable without long Spanish comments inside source files.

---

## Project purpose (do not lose sight)

This work must remain useful and honest for three audiences:

1. **Public policy / planning (Gobernación de Antioquia)** — conclusions must be scoped, cautious, and tied to data limitations. No overstated causal claims without design support.
2. **Econometrics 2 (coursework)** — reproducible pipelines, correct inference, transparent sample and model choices.
3. **Data science track** — **MLflow** will be integrated later; prefer scripts whose **parameters, inputs, and outputs** can be traced (paths, configs, run ids) without rewriting everything.

If a request would weaken validity for any of these, **say so** and suggest a better alternative.

---

## Data architecture (medallion)

| Layer   | Path            | Role |
|---------|-----------------|------|
| Raw     | `data/raw/`     | Source extracts; treat as immutable or version externally. |
| Bronze  | `data/bronze/`  | Landed data; minimal typing; preserve lineage. |
| Silver  | `data/silver/`  | Cleaned, harmonized keys (e.g. municipality codes), documented assumptions. |
| Gold    | `data/gold/`    | Analysis-ready panel, spatial weights metadata, model inputs. |

**Rule:** pipelines promote data **forward** through layers unless there is a documented exception (e.g. fixing a documented error in bronze with a clear audit note).

---

## Code layout

| Area         | Path                 | Content |
|-------------|----------------------|---------|
| ETL         | `src/pipelines/`     | `01_*` bronze, `02_*` silver, `03_*` gold — order of execution. |
| Exploration | `src/analysis/`      | EDA, maps, Moran / exploratory spatial analysis. |
| Models      | `src/models/`        | Spatial weights construction, SDM and related estimation. |
| Shared      | `src/common/`        | Reserved for shared helpers (paths, logging, I/O). |

**Outputs:** `results/` for tables and figures intended for reports; `logs/` for pipeline run logs.

Scripts under `src/pipelines/`, `src/analysis/`, and `src/models/` call `os.chdir` to the **repository root** so paths like `data/` and `logs/` resolve correctly when you run `python src/.../script.py` from the project root.

---

## Statistical and scientific rigor (non-negotiable)

- **Never invent** coefficients, p-values, file paths, variable names, or dataset shapes. If something is unknown, state what is missing and how to verify it (command, file, or code location).
- State **assumptions explicitly:** spatial weights specification, fixed vs random effects, sample period, missing-data handling, transformations.
- Prefer **robustness and sensitivity:** alternative `W`, subsamples, placebo-style checks when feasible; mention limits when data do not allow them.
- Separate **statistical significance** from **economic / policy relevance**; report uncertainty (standard errors, CIs) when reporting estimates.
- If variables are still **missing** for the research question, track them in the **Variable backlog** section below (or in an issue) — do not fabricate proxies.

---

## GitHub / professionalism

- Small, **focused** changes; avoid unrelated refactors.
- **Commit messages:** English; [Conventional Commits](https://www.conventionalcommits.org/) style encouraged (`feat:`, `fix:`, `docs:`, `refactor:`, etc.).
- Do **not** commit secrets, credentials, or large binary blobs that do not belong in git; respect `.gitignore`.
- **Push** when a logical unit of work is done or for backup — not on an arbitrary timer.

---

## Assistant behavior

- Ask clarifying questions **only** when **blocking** (ambiguous destructive action, missing critical data definition, or irreversible choice).
- Otherwise: state **short assumptions** and proceed; document them in code or commit message when non-obvious.
- Propose **concrete improvements** to structure and methods when they increase validity or reproducibility.
- **Explanation mode:** by default summarize by **logical blocks** (ingest → clean → merge → model → inference). Provide **line-by-line** walkthroughs only when the owner explicitly asks (e.g. “modo tutorial” / “explain every line in this function”).

---

## Execution and environment

- Prefer reproducible steps from the **repository root** and dependencies in `requirements.txt`.
- When suggesting commands, use paths consistent with this repo (`data/`, `src/`, `results/`, `logs/`).

---

## Variable backlog (owner-maintained)

*Add rows as the research design evolves. Assistants should not invent series not listed here without explicit owner approval.*

| Variable / theme | Intended source | Status | Notes |
|------------------|-----------------|--------|-------|
| *(example)*      | *(e.g. DANE)*   | planned | |

---

## Quick reference — existing pipeline IDs (illustrative)

Scripts and data layers evolve; **verify paths in the tree** before citing them in answers.

- Bronze / silver / gold: `src/pipelines/01_bronze_*.py`, `src/pipelines/02_silver_*.py`, `src/pipelines/03_gold_panel.py`
- Spatial EDA: `src/analysis/eda_spatial.py`
- Weights + SDM: `src/models/build_W_queen.py`, `src/models/spatial_model_sdm.py`
- Typical outputs: `data/gold/panel_gold.csv`, `data/gold/W_*`, `results/`, `logs/`

When this list drifts from reality, **update this section** in the same PR as the structural change.
