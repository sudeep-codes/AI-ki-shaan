# Design — Agri Chatbot

## Folder layout
```
agri-chatbot/
├── bot/
│   ├── handlers.py        # Telegram message/command handlers
│   ├── router.py          # intent routing (fertilizer / image / finance)
│   └── language.py        # detection + translation
├── rag/
│   ├── ingest.py           # builds the ChromaDB collections from /data
│   └── query.py            # retrieval function(s)
├── models/
│   └── disease_classifier.py  # image preprocessing + inference
├── finance/
│   └── schemes.py          # finance-specific prompt/formatting helpers
├── data/
│   ├── agronomy/            # ICAR + state advisory source docs
│   └── schemes/             # PM-KISAN, PMFBY, KCC docs
├── eval/
│   └── test_questions.json  # curated test set referenced in eval.md
├── .env.example
├── requirements.txt
└── README.md
```

## Sample conversation flows

### Happy path — fertilizer question (Hindi)
```
User: गेहूं के लिए कितनी यूरिया डालनी चाहिए?
Bot: [detects Hindi] → [translates to English for retrieval] →
     [retrieves ICAR wheat fertilizer passage] →
     [generates answer in Hindi, grounded in retrieved passage]
Bot: ICAR के दिशानिर्देशों के अनुसार, गेहूं के लिए बुवाई के समय
     लगभग 60-65 किग्रा/हेक्टेयर यूरिया... [continues, cites source]
```

### "I don't know" path
```
User: What fertilizer should I use for dragon fruit?
Bot: [RAG query returns low-similarity results, below threshold] →
Bot: I don't have reliable information on dragon fruit fertilizer
     in my current knowledge base. You may want to check with your
     local agriculture extension office.
```

### Low-confidence image path
```
User: [sends blurry photo of a leaf]
Bot: [classifier confidence 0.42, below 0.6 threshold] →
Bot: I'm not confident enough to diagnose this from the photo.
     Could you share a clearer close-up of the affected leaves,
     or describe what you're seeing (spots, wilting, discoloration)?
```

## Prompt templates

### Fertilizer/general RAG generation
```
System: You are an agricultural assistant for Indian farmers. 
Only use the CONTEXT below to answer. If the context does not 
contain enough information to answer confidently, say so clearly 
instead of guessing. Always mention which source the information 
comes from. Respond in {target_language}.

CONTEXT:
{retrieved_passages_with_source_names}

USER QUESTION:
{translated_question}
```

### Image diagnosis remedy generation
```
System: A plant disease classifier identified "{predicted_label}" 
with {confidence}% confidence. Using the CONTEXT below, explain 
what this is in simple terms and suggest a remedy. If the context 
doesn't cover this specific issue, say so. Respond in {target_language}.

CONTEXT:
{retrieved_remedy_passages}
```

### Finance/scheme generation
```
System: You are explaining Indian government agricultural finance 
schemes to a farmer. Only use the CONTEXT below. Explain eligibility 
and how to apply in plain, simple language — avoid bureaucratic 
jargon. If the context doesn't fully answer the question, say what's 
missing rather than inventing details. Respond in {target_language}.

CONTEXT:
{retrieved_scheme_passages}

USER QUESTION:
{translated_question}
```

## Open design questions (resolve during build, log the decision in memory.md)
- Keyword-based intent routing vs. a lightweight classifier — start with keywords, upgrade only if accuracy suffers
- Whether to show source citations to the end user directly, or only log them internally for your own evaluation (simpler UX if hidden, but transparency may matter for trust — worth testing both)
