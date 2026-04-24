import os
import sys
import warnings
warnings.filterwarnings("ignore")

os.chdir(r"C:\Users\Noel\Documents\Personal Projects\f1-rag-knowledge-assistant")
sys.path.insert(0, r"C:\Users\Noel\Documents\Personal Projects\f1-rag-knowledge-assistant")

from src.ingest import load_vector_store
from src.pipeline import build_rag_chain
from datasets import Dataset

test_questions = [
    "Who won the 2024 British Grand Prix?",
    "What was the fastest lap time recorded during the race by Lando Norris?",
    "Why did Tsunoda receive a penalty?",
    "What tyres did Norris use during the race?",
    "What was the weather like at Silverstone?"
]

# Load vector store and build chain
vector_store = load_vector_store()
qa_chain = build_rag_chain(vector_store)

# Run each question and collect results
answers = []
contexts = []

for question in test_questions:
    result = qa_chain.invoke(question)
    answers.append(result)
    docs = vector_store.similarity_search(question, k=5)
    contexts.append([doc.page_content for doc in docs])

# Simple manual evaluation — no OpenAI needed
print("\n" + "="*60)
print("RAG EVALUATION RESULTS")
print("="*60)

correct = 0
expected_keywords = [
    ["norris", "lando"], # Q1 — winner
    ["1:27.097", "1:25.912", "fastest", "norris"],       # Q2 — fastest lap
    ["tsunoda", "penalty", "yellow"],  # Q3 — penalty
    ["medium", "hard"],            # Q4 — tyres
    ["21", "overcast", "cloudy"],  # Q5 — weather
]

for i, (question, answer, keywords) in enumerate(zip(test_questions, answers, expected_keywords)):
    answer_lower = answer.lower()
    is_correct = any(kw in answer_lower for kw in keywords)
    if is_correct:
        correct += 1
    status = "✅ PASS" if is_correct else "❌ FAIL"
    print(f"\nQ{i+1}: {question}")
    print(f"A: {answer[:200]}...")
    print(f"Status: {status}")

accuracy = (correct / len(test_questions)) * 100
print("\n" + "="*60)
print(f"OVERALL ACCURACY: {correct}/{len(test_questions)} = {accuracy:.0f}%")
print("="*60)