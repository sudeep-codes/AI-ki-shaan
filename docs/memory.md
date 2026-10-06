# Memory — Agri Chatbot

Running log of decisions, reasoning, and things to revisit. Update this as you build — don't wait until the end, you'll forget the "why" behind decisions within days.

## Decisions log
*(format: date — decision — why)*

- [project start] — Chose Telegram over WhatsApp Business API — no business approval process, free, faster to ship an MVP
- [project start] — Chose ChromaDB over Pinecone — free, self-hosted, no external dependency at this scale
- [project start] — Separate RAG collections for agronomy vs. finance docs — avoids cross-domain retrieval noise

## Known issues / things to revisit
*(fill in as you hit them — e.g. "keyword routing misclassifies X type of question", "translation sounds unnatural for Y phrase")*

-

## Future ideas (explicitly out of MVP scope — see PRD.md)
- Voice input/output
- Real-time weather integration
- Mandi (market) price lookups
- More regional languages beyond the initial 2
- In-bot loan/scheme application flow (not just info)

## Source document inventory
*(track what you've added to /data and where it came from, so retrieval results are traceable and reproducible)*

| File | Source | Domain | Date added |
|---|---|---|---|
| | | | |

## Questions to ask yourself before each new phase
- Does this phase have a clear test from `phases.md`?
- Am I about to violate any rule in `rules.md`?
- Is this still in scope per `PRD.md`, or should it go in "Future ideas" above instead?
