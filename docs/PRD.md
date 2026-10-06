# PRD — Agri Chatbot

## Problem
Smallholder farmers often lack easy, trustworthy access to:
- Correct fertilizer type/quantity guidance for their specific crop and growth stage
- A fast way to identify what's wrong with their crop when something looks off
- Clear, plain-language info on government finance schemes (loans, insurance, subsidies)

Existing sources (govt PDFs, scattered advisories, word-of-mouth) are hard to search, often not in the farmer's language, and assume literacy/comfort with formal documents.

## Target user
- Smallholder/marginal farmer
- Has basic smartphone access and Telegram (or can have it installed for them)
- Comfortable typing short messages or sending photos in their own language
- Not assumed to read English or navigate government portals directly

## Core features (MVP scope)
1. **Fertilizer guidance** — ask about a crop + stage, get a grounded recommendation (type, quantity, timing)
2. **Crop issue diagnosis** — send a photo of an affected plant, get a likely diagnosis + suggested remedy
3. **Finance/scheme tips** — ask about loans, insurance, subsidies, get plain-language eligibility + how-to-apply info
4. **Multilingual support** — minimum 2 languages (English + Hindi, or swap in your second regional language), auto-detected

## Explicitly out of scope (for MVP)
- Voice input/output
- More than 2 languages
- Real-time weather or mandi (market) price integration
- Any transactional feature (applying for a loan *through* the bot)
- Persistent user accounts/history beyond the current chat session

## Success criteria
- Fertilizer Q&A: correctly answers at least 80% of a curated test set of ~20–30 crop/stage questions, in both supported languages
- Image diagnosis: correctly classifies at least 70% of a held-out test set from the chosen disease dataset
- Finance module: correctly summarizes eligibility/how-to-apply for at least PM-KISAN, PMFBY, and KCC without inventing details not in the source docs
- No hallucinated advice: every fertilizer/finance answer must be traceable to a specific source document (see `rules.md`)
- End-to-end demo: a non-technical person (not you) can message the bot in Hindi and get a correct, understandable answer without help

## Why this matters (for your own framing)
This isn't just "build a chatbot" — the hard part worth showcasing is grounding the LLM's answers in real sources and *measuring* how well that grounding holds up, especially for advice with real consequences (fertilizer, finance). That evaluation discipline is the differentiator for data/ML-focused roles.
