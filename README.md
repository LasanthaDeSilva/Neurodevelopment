# Convergence

A self-guided inquiry tool for adults exploring whether autism, ADHD, their
overlap, and twice-exceptionality (2e — giftedness alongside autism, ADHD, or
a learning disability) describe their lived experience. You rate a series of
statements, Convergence scores your answers across domains, and Gemini
provides a narrative interpretation you can ask follow-up questions about.

**Convergence is not a diagnostic instrument.** It's built to help you build
a clear, honest picture of your own patterns and have a better conversation
with a clinician — not to replace one. See [Disclaimer](#disclaimer) below.

## Features

- **Four modules** you can mix and match: Autism, ADHD, an Autism-vs-ADHD
  paired differential (unlocked when both are selected), and Giftedness & 2e.
- **~150 Likert items** (0 Never – 3 Always) across domains like social
  communication, masking, sensory sensitivity, executive function,
  attention, impulsivity, rejection sensitivity, and overexcitabilities —
  plus childhood history, onset/settings, and confounders (stress, sleep,
  substances, trauma, medical causes).
- **Transparent scoring**: domain bars, convergence tiers ("Strong" /
  "Moderate" / "Limited" / "Insufficient data"), and an autism-vs-ADHD lean —
  all explicitly framed as descriptive heuristics, not validated cutoffs.
- **AI analysis via Gemini**, grounded in your computed scores and item-level
  answers, with an interactive follow-up chat.
- **Save and resume**: download your progress as a `.json` file at any point
  and reload it later.
- **Export your results** as Markdown or a designed PDF (with a
  dependency-free fallback if `reportlab` isn't installed).
- **Nothing is stored server-side.** Your answers live only in your browser
  session; the only place data leaves your machine is the Gemini API call
  you trigger yourself.

## Requirements

- Python 3.10+
- [Streamlit](https://streamlit.io/) ≥ 1.40
- [`google-genai`](https://pypi.org/project/google-genai/) ≥ 1.40
- [ReportLab](https://pypi.org/project/reportlab/) ≥ 4.0 *(optional — PDF
  export falls back to a built-in, dependency-free generator if this isn't
  installed)*
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/apikey))

## Installation

```bash
pip install "streamlit>=1.40" "google-genai>=1.40" "reportlab>=4.0"
```

## Running

```bash
streamlit run nd.py
```

## API key

Convergence needs a Gemini API key to generate the AI analysis (the
questionnaire and scoring work without one). Provide it any of these ways:

1. **Environment variable**: `export GEMINI_API_KEY=your-key-here`
2. **Streamlit secrets**: add `GEMINI_API_KEY = "your-key-here"` to
   `.streamlit/secrets.toml`
3. **In-app**: paste it directly into the field on the results page (used
   only for that session, never stored)

## Model selection

Choose between three Gemini models in the app header. Each has an
environment-variable override if you want to pin a different model ID
without editing the code:

| App label | Default model ID | Override variable |
|---|---|---|
| Gemini 3.1 Pro | `gemini-3.1-pro-preview` | `GEMINI_MODEL_PRO` |
| Gemini 3.6 Flash | `gemini-3.6-flash` | `GEMINI_MODEL_FLASH` |
| Gemini 3.5 Flash-Lite | `gemini-3.5-flash-lite` | `GEMINI_MODEL_LITE` |

## How it's structured

Everything lives in the single file `nd.py`:

- **Questionnaire data** — items, sections, and modules, defined as typed
  dataclasses.
- **Scoring** — converts raw answers into per-domain stats, convergence
  tiers, the autism/ADHD differential, and the 2e summary.
- **Prompts** — the system prompt and report template sent to Gemini,
  including instructions to cite item IDs and avoid diagnostic language.
- **Gemini integration** — streaming chat completion plus a follow-up chat
  loop.
- **App / pages** — the Streamlit UI: intro, one page per question section,
  and the results page.
- **PDF export** — a styled ReportLab build, with a hand-rolled,
  dependency-free fallback PDF writer if ReportLab is unavailable.

## Privacy

Nothing is persisted on a server. Your answers exist only in your browser's
session state for the lifetime of that tab. Use **Save progress** in the
header to download a `.json` file you can upload later (via "Resume from a
saved file" on the intro page) to pick up where you left off — if you close
the tab without saving, that progress is gone. Generating an AI analysis
sends your scores, item responses, and (optionally) your free-text notes to
Google's Gemini API using the key you provide.

## Disclaimer

Convergence supports self-understanding; it is not a diagnostic instrument.
Self-identification is a valid endpoint for many people, but accommodations
or treatment generally require a formal evaluation. Validated screeners
(AQ, RAADS-R, CAT-Q, ASRS v1.1) are referenced throughout but are themselves
screening tools, not diagnostics. If you're in distress or thinking about
harming yourself, contact your local emergency number or a crisis line
(988 in the US; [findahelpline.com](https://findahelpline.com) elsewhere).

## Credit

Designed & Engineered by Nikolai de Silva · © 2026 ninolades.com
