import asyncio
import json
import os
import sys
from datetime import datetime

# Ensure stdout handles unicode (Hindi text) on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.chatbot_pipeline import get_chatbot_pipeline
from finance.scheme_router import detect_intent
from config import settings

EVAL_FILE = os.path.join(os.path.dirname(__file__), "test_questions.json")

def get_llm_client():
    pass



EVAL_PROMPT_TEMPLATE = """
You are an expert evaluator grading a conversational AI assistant for Indian farmers.
Evaluate the following interaction:

Question: {question}
Expected topics/keywords: {expected_topics}
Assistant Answer: {answer}
Sources provided: {sources}

Check if the Assistant Answer:
1. Addresses the core question accurately.
2. Contains the expected topics/keywords (or their semantic equivalents in the correct language).
3. If the question is out-of-domain (like cricket), the assistant should gracefully decline (e.g. "I do not know" or "I can only help with farming").
4. If sources are provided, ensure the answer doesn't hallucinate facts not supported by a typical knowledge base on this topic.

Respond with ONLY a JSON object containing:
{{
  "pass": true or false,
  "reason": "short explanation of why it passed or failed"
}}
"""

async def evaluate_question(item, chatbot, llm_client):
    question = item["question"]
    expected = ", ".join(item["expected_topics"])
    module = item["module"]
    
    print(f"\n[{module.upper()}] Evaluating: {question}")
    
    # 1. Detect intent and process
    intent = detect_intent(question)
    result = await chatbot.process_telegram_message(question, intent=intent)
    
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    
    # 2. Call LLM to grade
    prompt = EVAL_PROMPT_TEMPLATE.format(
        question=question,
        expected_topics=expected,
        answer=answer,
        sources=json.dumps(sources, ensure_ascii=False)
    )
    
    import httpx
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={settings.GEMINI_API_KEY}"
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            res = await http_client.post(url, json=payload)
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "").strip()
            else:
                print(f"Error from Gemini: {res.text}")
                text = ""
                
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:-3]
            
        grade = json.loads(text)
        is_pass = grade.get("pass", False)
        reason = grade.get("reason", "")
    except Exception as e:
        print(f"Error parsing LLM grade: {e}")
        is_pass = False
        reason = "Failed to parse evaluation response."
        
    print(f"  -> Result: {'PASS' if is_pass else 'FAIL'}")
    print(f"  -> Reason: {reason}")
    
    return {
        "id": item["id"],
        "module": module,
        "question": question,
        "language": item["language"],
        "answer": answer,
        "pass": is_pass,
        "reason": reason
    }

async def run_evaluations():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    chatbot = get_chatbot_pipeline()
    llm_client = get_llm_client()
    
    results = []
    passed = 0
    
    for q in questions:
        res = await evaluate_question(q, chatbot, llm_client)
        results.append(res)
        if res["pass"]:
            passed += 1
        
        print("Sleeping for 15s to respect rate limits...")
        await asyncio.sleep(15)
            
    total = len(questions)
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Score: {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Generate markdown report
    report = f"## Automated Evaluation Run ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n"
    report += f"**Score:** {passed}/{total} ({passed/total*100:.1f}%)\n\n"
    
    report += "| # | Module | Question | Lang | Pass/Fail | Notes |\n"
    report += "|---|---|---|---|---|---|\n"
    for r in results:
        status = "✅ PASS" if r["pass"] else "❌ FAIL"
        report += f"| {r['id']} | {r['module']} | {r['question']} | {r['language']} | {status} | {r['reason']} |\n"
        
    report_path = os.path.join(os.path.dirname(__file__), "eval_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"Detailed report saved to {report_path}")

if __name__ == "__main__":
    asyncio.run(run_evaluations())
