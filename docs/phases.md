# Phases — Agri Chatbot

Each phase should be independently testable before moving to the next (see `rules.md` #12). Rough time estimates assume ~10–15 hrs/week solo.

## Phase 1 — Project scaffolding
**Depends on:** nothing
**Output:** runnable empty project with folder structure, requirements.txt, .env.example, README
**Test:** project installs cleanly and the Telegram bot responds to `/start`
**Est:** 2–3 hrs

## Phase 2 — Knowledge base ingestion (RAG)
**Depends on:** Phase 1; manual collection of source docs (ICAR fertilizer guidelines, state advisories) into `/data`
**Output:** `ingest.py` and `query.py` in `/rag`, working ChromaDB collection
**Test:** querying a known fact (e.g. "wheat fertilizer dosage") returns the correct source passage
**Est:** 4–6 hrs coding + variable time for source doc collection (budget separately, this is research time not coding time)

## Phase 3 — Core chatbot logic (fertilizer Q&A)
**Depends on:** Phase 2
**Output:** end-to-end text flow: question → retrieval → grounded Gemini answer
**Test:** 10 sample fertilizer questions answered correctly and cited (see `eval.md`)
**Est:** 4–5 hrs

## Phase 4 — Regional language support
**Depends on:** Phase 3
**Output:** language detection + translation layered into the Phase 3 flow, `/language` command
**Test:** same 10 questions, now asked in Hindi (or chosen second language), answered correctly in that language
**Est:** 3–4 hrs

## Phase 5 — Crop issue diagnosis from photos
**Depends on:** Phase 2 (reuses agronomy RAG for remedies), Phase 4 (for language-aware replies)
**Output:** photo handling, classifier integration, confidence threshold logic
**Test:** held-out test images classified correctly; low-confidence images correctly trigger a clarifying question instead of a guess
**Est:** 5–7 hrs

## Phase 6 — Finance & scheme tips module
**Depends on:** Phase 2 pattern (separate RAG collection), Phase 4
**Output:** `/finance` command, intent detection for loan/scheme/insurance questions, separate scheme-docs RAG collection
**Test:** correct eligibility/how-to-apply summaries for PM-KISAN, PMFBY, KCC, in both languages
**Est:** 4–5 hrs + scheme doc collection time

## Phase 7 — Testing & evaluation
**Depends on:** Phases 3–6 complete
**Output:** populated `eval.md` with full test results across all modules and both languages
**Test:** this phase *is* the test — produces the pass/fail numbers referenced in `PRD.md` success criteria
**Est:** 4–6 hrs

## Phase 8 — Deployment
**Depends on:** Phase 7 passing at acceptable rates
**Output:** hosted, publicly reachable bot (Render or HF Spaces), webhook instead of local polling
**Test:** a friend who hasn't seen the code can message the live bot and get correct answers
**Est:** 2–4 hrs

---
**Total rough estimate:** 3–5 weeks at 10–15 hrs/week, not counting source-document research time, which is hard to bound in advance.
