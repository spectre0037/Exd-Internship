import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


model_names = [
    'all-MiniLM-L6-v2',
    'all-mpnet-base-v2',
    'BAAI/bge-small-en-v1.5'
]


sample_text = "FastAPI handles asynchronous routing efficiently."

print(" Starting Model Benchmark Experiment...\n")


results = []

for name in model_names:
    print(f" Loading/Downloading model: {name}...")
    model = SentenceTransformer(name)
    start_time = time.perf_counter()
    embedding = model.encode(sample_text)
    end_time = time.perf_counter()
    execution_time_ms = (end_time - start_time) * 1000
    vector_dimension = embedding.shape[0]
    
    results.append({
        "Model Name": name,
        "Vector Dimension": vector_dimension,
        "Encoding Time (ms)": round(execution_time_ms, 2)
    })

df = pd.DataFrame(results)
print("\n📊 Benchmark Experiment Results:")
print(df.to_string(index=False))