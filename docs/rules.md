# Rules — Agri Chatbot

These are hard constraints for both the AI agent building this (Antigravity) and the chatbot's own runtime behavior. Keep this file open/referenced throughout the build — it's the thing most likely to get silently violated by "helpful" generated code.

## Grounding & honesty
1. **Never answer fertilizer or finance questions outside retrieved context.** If the RAG query returns nothing relevant (low similarity scores), the bot must say it doesn't have reliable information on that, rather than letting the LLM fill the gap from general knowledge. Wrong fertilizer dosage or wrong scheme eligibility info has real consequences.
2. **Every fertilizer/finance answer must cite which source document it came from** (e.g. "Based on [ICAR wheat guideline]..."). This is both a trust signal for the user and a debugging aid for you — if an answer looks wrong, you can trace it to the source chunk.
3. **The image diagnosis must state its confidence** and, below a set threshold (e.g. 0.6), ask a clarifying question ("Can you share a clearer photo, or describe the symptoms?") instead of asserting a diagnosis.
4. System prompt for the generation step must explicitly instruct: "Only use the provided context to answer. If the context doesn't cover the question, say so clearly instead of guessing."

## Language
5. The bot always replies in the same language the user wrote in, detected per-message (not just set once) — a user may switch languages mid-conversation.
6. Translation happens for retrieval purposes only; the final reply is generated/phrased naturally in the target language, not machine-translated word-for-word from an English draft where avoidable.

## Engineering
7. No hardcoded API keys or secrets anywhere in the codebase — all via `.env`, with `.env.example` committed instead of the real file.
8. Every external call (Gemini API, Hugging Face model, Telegram API) must have basic error handling — a failed call should produce a graceful fallback message, not a crash or a silent hang.
9. Keep the fertilizer/agronomy and finance/scheme RAG collections separate (see `architecture.md`) — don't let one query pull irrelevant results from the other domain.
10. Log every (question, retrieved sources, final answer) triple during development — this log is the raw material for `eval.md`.

## Scope discipline
11. Don't add features not in `PRD.md`'s MVP scope mid-build (e.g. weather integration, voice input) — note them in `memory.md` as future ideas instead, so the agent doesn't scope-creep the current phase.
12. Each phase in `phases.md` should be testable on its own before moving to the next — don't let the agent chain multiple phases into one unverified commit.
