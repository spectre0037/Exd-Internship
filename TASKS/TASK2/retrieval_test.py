import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "FastAPI utilizes Pydantic for data validation and serialization.",
    "Asynchronous code in Python uses the async and await keywords.",
    "Uvicorn is a lightning-fast ASGI server implementation, using uvloop.",
    "Dependency injection in FastAPI allows for easy database session management.",
    "Flask is a WSGI microframework, whereas FastAPI is an ASGI framework.",
    "To handle database migrations in Python, Alembic is the industry standard.",
    "CORS or Cross-Origin Resource Sharing must be configured for frontend apps.",
    "Middlewares in FastAPI execute before a request reaches the path operation.",
    "Background tasks can be used in FastAPI to run heavy jobs after returning a response.",
    "SQLAlchemy is an Object-Relational Mapper (ORM) used to interact with databases.",
    "Docker containers isolate Python applications with all their dependencies.",
    "JSON Web Tokens (JWT) are commonly used for securing API endpoints.",
    "The startup event handler in FastAPI is ideal for initializing database pools.",
    "Redis is an in-memory data structure store used as a database, cache, and message broker.",
    "Pytest is a robust framework that makes it easy to write small, readable tests."
]

queries = [
    "How do I handle asynchronous tasks in the background?",
    "What tool should I use for database schema migrations?",
    "How do I secure my backend routes?"
]

model_names = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'BAAI/bge-small-en-v1.5']

print("🚀 Starting Mini Retrieval Test...\n")

for model_name in model_names:
    print("=" * 60)
    print(f"🤖 EVALUATING MODEL: {model_name}")
    print("=" * 60)
    
    model = SentenceTransformer(model_name)
    doc_embeddings = model.encode(documents)
    
    for query in queries:
        query_embedding = model.encode([query])
        
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        top_3_indices = np.argsort(similarities)[::-1][:3]
        
        print(f"\n🔍 Query: '{query}'")
        print("📋 Top 3 Retrieved Results:")
        for rank, idx in enumerate(top_3_indices, 1):
            score = similarities[idx]
            print(f"  {rank}. [Score: {score:.4f}] {documents[idx]}")
    print("\n")