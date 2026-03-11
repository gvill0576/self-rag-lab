import boto3
import json
from typing import List, Dict
import chromadb

# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

print("✅ Bedrock client initialized!")

# Sample ML documents
DOCUMENTS = [
    {
        "id": "doc1",
        "content": "Neural networks learn through backpropagation. The algorithm calculates gradients of the loss function with respect to each weight and adjusts them to minimize error."
    },
    {
        "id": "doc2",
        "content": "Overfitting occurs when a model memorizes training data instead of learning patterns. It performs well on training data but poorly on new, unseen data."
    },
    {
        "id": "doc3",
        "content": "To prevent overfitting, use techniques like regularization (L1/L2), dropout, early stopping, or collect more training data."
    },
    {
        "id": "doc4",
        "content": "Python is a popular programming language used for web development, data science, and automation."
    },
    {
        "id": "doc5",
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    },
    {
        "id": "doc6",
        "content": "Learning rate controls how much to adjust weights during training. Too high causes instability, too low causes slow convergence."
    }
]


def create_embedding(text: str) -> List[float]:
    """Create embedding using Bedrock Titan"""
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text})
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def setup_vectorstore():
    """Create ChromaDB vector store with documents"""
    client = chromadb.Client()

    collection = client.create_collection(
        name="ml_docs",
        metadata={"hnsw:space": "cosine"}
    )

    print("📚 Creating embeddings...")
    for doc in DOCUMENTS:
        embedding = create_embedding(doc["content"])
        collection.add(
            ids=[doc["id"]],
            embeddings=[embedding],
            documents=[doc["content"]],
            metadatas=[{"source": doc["id"]}]
        )
        print(f"   Added: {doc['id']}")

    print(f"✅ Vector store ready with {len(DOCUMENTS)} documents\n")
    return collection


# Initialize vector store at module level
vectorstore = setup_vectorstore()


def invoke_bedrock(prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
    """Helper to invoke Bedrock"""
    response = bedrock.converse(
        modelId="us.amazon.nova-lite-v1:0",
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}
    )
    return response["output"]["message"]["content"][0]["text"]


def retrieve(query: str, k: int = 3) -> List[Dict]:
    """Basic retrieval from vector store"""
    query_embedding = create_embedding(query)

    results = vectorstore.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    documents = []
    for i, doc in enumerate(results["documents"][0]):
        documents.append({
            "content": doc,
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i]
        })

    return documents


def basic_rag(query: str) -> Dict:
    """Basic RAG without validation"""
    docs = retrieve(query, k=3)
    context = "\n\n".join([d["content"] for d in docs])

    prompt = f"""Answer the question based on the context provided.

Context:
{context}

Question: {query}

Answer:"""

    answer = invoke_bedrock(prompt)

    return {
        "answer": answer,
        "documents": docs,
        "validated": False
    }


def validate_relevance(query: str, document: str) -> Dict:
    """Check if a document is relevant to the query"""

    prompt = f"""Evaluate if this document is relevant to answering the question.

Question: {query}

Document: {document}

Respond in this exact format:
RELEVANT: [YES or NO]
REASON: [One sentence explanation]"""

    response = invoke_bedrock(prompt, max_tokens=100, temperature=0.1)

    is_relevant = "YES" in response.upper().split("RELEVANT:")[1].split("\n")[0] if "RELEVANT:" in response else False
    reason = response.split("REASON:")[1].strip() if "REASON:" in response else "Could not parse"

    return {"relevant": is_relevant, "reason": reason}


def retrieve_with_validation(query: str, k: int = 3) -> Dict:
    """Retrieve and validate document relevance"""

    candidates = retrieve(query, k=k * 2)

    validated = []
    rejected = []

    print(f"\n🔍 Validating {len(candidates)} candidates...")

    for doc in candidates:
        result = validate_relevance(query, doc["content"])

        if result["relevant"]:
            validated.append(doc)
            print(f"   ✅ {doc['source']}: Relevant")
        else:
            rejected.append({"doc": doc, "reason": result["reason"]})
            print(f"   ❌ {doc['source']}: {result['reason'][:50]}...")

        if len(validated) >= k:
            break

    return {
        "documents": validated,
        "rejected": rejected,
        "validation_rate": len(validated) / len(candidates) if candidates else 0
    }


def validate_grounding(query: str, context: str, answer: str) -> Dict:
    """Check if the answer is grounded in the context"""

    prompt = f"""Evaluate if this answer is fully supported by the context provided.

Context:
{context}

Question: {query}

Answer: {answer}

Respond in this exact format:
GROUNDED: [FULLY, PARTIALLY, or NOT]
UNSUPPORTED: [List any claims not in context, or "None"]
CONFIDENCE: [HIGH, MEDIUM, or LOW]"""

    response = invoke_bedrock(prompt, max_tokens=200, temperature=0.1)

    grounded_line = response.split("GROUNDED:")[1].split("\n")[0] if "GROUNDED:" in response else ""
    fully_grounded = "FULLY" in grounded_line.upper()
    partially_grounded = "PARTIALLY" in grounded_line.upper()

    confidence_line = response.split("CONFIDENCE:")[1].split("\n")[0] if "CONFIDENCE:" in response else "MEDIUM"
    confidence = "HIGH" if "HIGH" in confidence_line else "LOW" if "LOW" in confidence_line else "MEDIUM"

    return {
        "fully_grounded": fully_grounded,
        "partially_grounded": partially_grounded,
        "confidence": confidence,
        "raw": response
    }


def self_rag(query: str) -> Dict:
    """Self-RAG with relevance and grounding validation"""

    print(f"\n🔍 Self-RAG Query: {query}")
    print("-" * 50)

    retrieval = retrieve_with_validation(query, k=3)

    if not retrieval["documents"]:
        return {
            "answer": "I couldn't find relevant information to answer this question.",
            "confidence": "LOW",
            "documents_used": 0,
            "documents_rejected": len(retrieval["rejected"]),
            "fully_grounded": False,
            "validation_rate": 0
        }

    context = "\n\n".join([d["content"] for d in retrieval["documents"]])

    generation_prompt = f"""Answer the question based only on the context provided.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""

    answer = invoke_bedrock(generation_prompt)

    print("\n🔍 Validating answer grounding...")
    grounding = validate_grounding(query, context, answer)

    grounding_status = "✅ Fully grounded" if grounding["fully_grounded"] else \
                       "⚠️ Partially grounded" if grounding["partially_grounded"] else \
                       "❌ Not grounded"
    print(f"   {grounding_status}")
    print(f"   Confidence: {grounding['confidence']}")

    if grounding["fully_grounded"] and retrieval["validation_rate"] >= 0.5:
        confidence = "HIGH"
    elif grounding["partially_grounded"] or retrieval["validation_rate"] >= 0.3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "answer": answer,
        "confidence": confidence,
        "documents_used": len(retrieval["documents"]),
        "documents_rejected": len(retrieval["rejected"]),
        "fully_grounded": grounding["fully_grounded"],
        "validation_rate": retrieval["validation_rate"]
    }


def rewrite_query(query: str) -> str:
    """Rewrite query for better retrieval"""

    prompt = f"""The search query below returned poor results.
Rewrite it to be more specific and likely to match technical documentation.

Original query: {query}

Return ONLY the rewritten query, nothing else."""

    return invoke_bedrock(prompt, max_tokens=50, temperature=0.7).strip()


def corrective_retrieve(query: str, min_relevant: int = 2) -> Dict:
    """Retrieve with correction when results are poor"""

    result = retrieve_with_validation(query, k=4)

    if len(result["documents"]) >= min_relevant:
        return {**result, "corrected": False}

    print(f"\n⚠️ Only {len(result['documents'])} relevant docs. Attempting correction...")

    rewritten = rewrite_query(query)
    print(f"📝 Rewritten: {rewritten}")

    result_retry = retrieve_with_validation(rewritten, k=4)

    seen = set()
    combined = []

    for doc in result["documents"] + result_retry["documents"]:
        if doc["content"] not in seen:
            combined.append(doc)
            seen.add(doc["content"])

    return {
        "documents": combined,
        "rejected": result["rejected"] + result_retry["rejected"],
        "corrected": True,
        "original_query": query,
        "rewritten_query": rewritten,
        "validation_rate": len(combined) / max(len(result["documents"]) + len(result_retry["documents"]), 1)
    }


class ConversationalSelfRAG:
    """Self-RAG with conversation memory"""

    def __init__(self):
        self.history = []
        self.max_history = 5

    def expand_with_context(self, query: str) -> str:
        """Expand query using conversation context"""

        if not self.history:
            return query

        last = self.history[-1]

        prompt = f"""Given this conversation context, rewrite the current query to be self-contained.

Previous Question: {last['query']}
Previous Answer: {last['answer'][:200]}

Current Question: {query}

If the current question references the previous exchange (using "it", "that", "this"),
rewrite it to be standalone. Otherwise, return the original question.

Rewritten Question:"""

        expanded = invoke_bedrock(prompt, max_tokens=100, temperature=0.1).strip()

        if len(expanded) > len(query) * 3:
            return query

        return expanded

    def summarize_history(self) -> str:
        """Summarize conversation history to save tokens"""

        if len(self.history) <= 2:
            return "\n".join([
                f"Q: {h['query']}\nA: {h['answer'][:150]}..."
                for h in self.history
            ])

        old_turns = self.history[:-2]
        recent_turns = self.history[-2:]

        old_text = "\n".join([
            f"Q: {h['query']}\nA: {h['answer'][:100]}"
            for h in old_turns
        ])

        summary_prompt = f"""Summarize this conversation in 2-3 sentences:

{old_text}

Summary:"""

        summary = invoke_bedrock(summary_prompt, max_tokens=100, temperature=0.3)

        recent_text = "\n".join([
            f"Q: {h['query']}\nA: {h['answer'][:150]}"
            for h in recent_turns
        ])

        return f"Earlier: {summary}\n\nRecent:\n{recent_text}"

    def chat(self, query: str) -> Dict:
        """Process a query with memory"""

        print(f"\n💬 User: {query}")

        expanded = self.expand_with_context(query)
        if expanded != query:
            print(f"📝 Expanded: {expanded}")

        retrieval = corrective_retrieve(expanded)

        if not retrieval["documents"]:
            answer = "I couldn't find relevant information to answer this question."
            confidence = "LOW"
        else:
            doc_context = "\n\n".join([d["content"] for d in retrieval["documents"]])
            history_context = self.summarize_history() if self.history else ""

            prompt = f"""You are a helpful assistant. Use the conversation history and retrieved context to answer.

Conversation History:
{history_context}

Retrieved Context:
{doc_context}

Current Question: {query}

Answer:"""

            answer = invoke_bedrock(prompt)

            grounding = validate_grounding(query, doc_context, answer)
            confidence = "HIGH" if grounding["fully_grounded"] else "MEDIUM"

        self.history.append({"query": query, "answer": answer})

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        print(f"🤖 Assistant: {answer}")
        print(f"📊 Confidence: {confidence}")

        return {
            "answer": answer,
            "confidence": confidence,
            "expanded_query": expanded,
            "history_length": len(self.history)
        }

    def clear(self):
        """Clear conversation history"""
        self.history = []
        print("🗑️ History cleared")


def compare_approaches(query: str):
    """Compare basic RAG vs Self-RAG"""
    print("\n" + "=" * 70)
    print(f"📊 COMPARING APPROACHES: '{query}'")
    print("=" * 70)

    print("\n❌ BASIC RAG (No Validation)")
    print("-" * 50)
    basic_result = basic_rag(query)
    print(f"Answer: {basic_result['answer'][:200]}...")
    print(f"Documents: {len(basic_result['documents'])}")
    print(f"Validated: {basic_result['validated']}")

    print("\n✅ SELF-RAG (With Validation)")
    print("-" * 50)
    self_result = self_rag(query)
    print(f"Answer: {self_result['answer'][:200]}...")
    print(f"Documents used: {self_result['documents_used']}")
    print(f"Documents rejected: {self_result['documents_rejected']}")
    print(f"Confidence: {self_result['confidence']}")
    print(f"Fully grounded: {self_result['fully_grounded']}")

    print("\n" + "=" * 70)


def test_basic_rag():
    print("=" * 60)
    print("🔍 BASIC RAG (No Validation)")
    print("=" * 60)

    query = "How do neural networks learn?"
    result = basic_rag(query)

    print(f"\n❓ Query: {query}")
    print(f"\n📚 Retrieved {len(result['documents'])} documents:")
    for doc in result["documents"]:
        print(f"   - [{doc['source']}] {doc['content'][:60]}...")
    print(f"\n💬 Answer: {result['answer']}")


def test_relevance_validation():
    print("\n" + "=" * 60)
    print("🔍 TESTING RELEVANCE VALIDATION")
    print("=" * 60)

    query = "How do neural networks learn?"
    result = retrieve_with_validation(query, k=3)

    print(f"\n📊 Validation Rate: {result['validation_rate']:.0%}")
    print(f"✅ Validated: {len(result['documents'])}")
    print(f"❌ Rejected: {len(result['rejected'])}")


def test_self_rag():
    print("\n" + "=" * 60)
    print("🧠 SELF-RAG PIPELINE")
    print("=" * 60)

    queries = [
        "How do neural networks learn?",
        "What causes overfitting and how do I prevent it?",
        "What's the weather today?"
    ]

    for query in queries:
        result = self_rag(query)

        print(f"\n💬 Answer: {result['answer'][:200]}...")
        print(f"📊 Confidence: {result['confidence']}")
        print(f"📚 Documents: {result['documents_used']} used, {result.get('documents_rejected', 0)} rejected")
        print("-" * 50)


def test_corrective_retrieval():
    print("\n" + "=" * 60)
    print("🔄 CORRECTIVE RETRIEVAL")
    print("=" * 60)

    query = "why is my model bad?"

    result = corrective_retrieve(query)

    print(f"\n📊 Results:")
    print(f"   Corrected: {result['corrected']}")
    if result['corrected']:
        print(f"   Original: {result['original_query']}")
        print(f"   Rewritten: {result['rewritten_query']}")
    print(f"   Documents found: {len(result['documents'])}")


def test_conversational_memory():
    print("\n" + "=" * 60)
    print("💬 CONVERSATIONAL MEMORY")
    print("=" * 60)

    rag = ConversationalSelfRAG()

    rag.chat("What is overfitting?")
    print()
    rag.chat("How do I prevent it?")
    print()
    rag.chat("What about neural networks?")
    print()
    rag.chat("How do they learn?")


if __name__ == "__main__":
    print("🚀 Self-RAG Lab")
    print("=" * 60)

    compare_approaches("How do neural networks learn?")
    compare_approaches("What's the weather today?")

    print("\n")
    test_conversational_memory()

    print("\n✅ Lab complete!")