"""
Simple test to check agent functionality
"""

from solution import ContestantAgent
import time

API_KEY = "tpsg-7C0gRSRQbimicxdfnfNMigsTZ7gTPjY"

print("Initializing agent...")
agent = ContestantAgent(API_KEY)

# Test 1: Simple calculation
print("\n" + "="*60)
print("TEST 1: Simple Math")
print("="*60)
start = time.time()
problem1 = {"problem": "What is 10 + 15?"}
answer1 = agent.solve_lock(problem1, [])
elapsed = time.time() - start
print(f"Problem: What is 10 + 15?")
print(f"Answer: {answer1}")
print(f"Time: {elapsed:.2f}s")

# Test 2: Persian calculation
print("\n" + "="*60)
print("TEST 2: Persian Math")
print("="*60)
start = time.time()
problem2 = {"problem": "دو به علاوه دو"}
answer2 = agent.solve_lock(problem2, [])
elapsed = time.time() - start
print(f"Problem: دو به علاوه دو")
print(f"Answer: {answer2}")
print(f"Time: {elapsed:.2f}s")

# Test 3: CSV row count (real URL)
print("\n" + "="*60)
print("TEST 3: CSV Row Count")
print("="*60)
start = time.time()
problem3 = {
    "problem": "how many rows are there in the provided CSV file?\nhttps://gist.githubusercontent.com/rnirmal/e01acfdaf54a6f9b24e91ba4cae63518/raw/6b589a5c5a8517ad2bfc7b1c0da3da2eae47c3bb/books.csv"
}
answer3 = agent.solve_lock(problem3, [])
elapsed = time.time() - start
print(f"Problem: Count rows in CSV")
print(f"Answer: {answer3}")
print(f"Expected: Around 61")
print(f"Time: {elapsed:.2f}s")

# Test 4: With history
print("\n" + "="*60)
print("TEST 4: Learning from History")
print("="*60)
start = time.time()
history = [
    ("Paris", "Wrong! That's the capital of France, not England.")
]
problem4 = {"problem": "What is the capital of England?"}
answer4 = agent.solve_lock(problem4, history)
elapsed = time.time() - start
print(f"Problem: What is the capital of England?")
print(f"Previous wrong answer: Paris (was told it's France's capital)")
print(f"Answer: {answer4}")
print(f"Time: {elapsed:.2f}s")

print("\n" + "="*60)
print("ALL TESTS COMPLETED!")
print("="*60)
