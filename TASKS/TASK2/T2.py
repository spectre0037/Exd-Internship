import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1. Define the models we want to evaluate
model_names = [
    'all-MiniLM-L6-v2',
    'all-mpnet-base-v2',
    'BAAI/bge-small-en-v1.5'
]

# 2. Define a standard domain-specific sample sentence
sample_text = "FastAPI handles asynchronous routing efficiently."

print("🚀 Starting Model Benchmark Experiment...\n")

# Dictionary to hold our results for analysis
results = []

for name in model_names:
    print(f"📥 Loading/Downloading model: {name}...")
    # This automatically downloads the model weights on the first run
    model = SentenceTransformer(name)
    
    # Measure execution time for encoding
    start_time = time.perf_counter()
    embedding = model.encode(sample_text)
    end_time = time.perf_counter()
    
    # Calculate execution time in milliseconds
    execution_time_ms = (end_time - start_time) * 1000
    vector_dimension = embedding.shape[0]
    
    results.append({
        "Model Name": name,
        "Vector Dimension": vector_dimension,
        "Encoding Time (ms)": round(execution_time_ms, 2)
    })

# 3. Display the benchmark results in a clean table
df = pd.DataFrame(results)
print("\n📊 Benchmark Experiment Results:")
print(df.to_string(index=False))