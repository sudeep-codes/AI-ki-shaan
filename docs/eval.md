# Eval — Agri Chatbot

This is what turns "I built a chatbot" into a measurable, defensible result — fill it in as you complete Phases 3, 5, and 6. It's also your strongest resume material: concrete accuracy numbers beat a demo video.

## Fertilizer Q&A evaluation

Test set: ~20–30 curated crop/stage questions, covering both supported languages.

| # | Question | Language | Expected answer (source) | Bot's answer | Pass/Fail | Notes |
|---|---|---|---|---|---|---|
| 1 | What is the recommended NPK dose for rice? | English | Nitrogen, Phosphorus, Potassium, amount, kg/ha | To be evaluated (Pipeline setup) | - | Setup complete, waiting on run |
| 2 | गेहूं की फसल में यूरिया कब डालना चाहिए? | Hindi | यूरिया, सिंचाई, दिन, अवस्था | To be evaluated | - | - |
| 3 | How to control aphids in cotton? | English | insecticide, spray, neem oil, chemical | To be evaluated | - | - |
| 4 | गन्ने में लाल सड़न (Red rot) रोग का क्या उपाय है? | Hindi | बीजोपचार, स्वस्थ बीज, फफूंदनाशक | To be evaluated | - | - |
| 5 | Can I grow apples in a hot desert climate? | English | temperature, chilling hours, unsuitable | To be evaluated | - | - |

**Summary:** Pending automated run `python eval/run_evals.py`

## Image diagnosis evaluation

Test set: held-out images from the chosen disease dataset (not used in any prompt/few-shot examples), plus a few deliberately ambiguous/blurry ones to test the confidence-threshold behavior.

| # | Image | True label | Predicted label | Confidence | Correct? | Threshold behavior correct? |
|---|---|---|---|---|---|---|
| 1 | | | | | | |

**Summary:** ___ / ___ correct (___%); ___ / ___ low-confidence cases correctly deferred

## Finance/scheme evaluation

Test set: questions covering PM-KISAN, PMFBY, KCC eligibility and application process, both languages.

| # | Question | Language | Expected info present | Bot's answer accurate? | No invented details? | Notes |
|---|---|---|---|---|---|---|
| 1 | What are the eligibility criteria for the PM-KISAN scheme? | English | landholding, farmers, exclusions, institutions | To be evaluated | - | - |
| 2 | PMFBY (प्रधानमंत्री फसल बीमा योजना) के तहत कौन सी फसलें कवर की जाती हैं? | Hindi | खरीफ, रबी, वाणिज्यिक, बागवानी | To be evaluated | - | - |
| 3 | How much financial assistance is provided per year under PM-KISAN? | English | 6000, installments, 2000 | To be evaluated | - | - |
| 4 | KCC (किसान क्रेडिट कार्ड) पर ब्याज दर क्या है? | Hindi | ब्याज दर, 7%, सबवेंशन, 4% | To be evaluated | - | - |

**Summary:** Pending automated run `python eval/run_evals.py`

## Groundedness spot-check
Pick 10 random answers across all modules and manually verify every claim traces back to an actual retrieved source passage (not the LLM's general knowledge). This is the check that matters most given the rules in `rules.md` #1–2.

- Spot-checked: ___ / 10
- Fully grounded: ___ / 10
- Issues found:

## What you'd change at scale
*(fill in after evaluation — this section is what shows depth of thinking beyond "it works")*
- 
