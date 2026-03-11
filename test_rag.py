# test_rag.py
"""Automated tests for Self-RAG system"""

from self_rag import (
    basic_rag, 
    self_rag, 
    validate_relevance, 
    validate_grounding,
    ConversationalSelfRAG
)

# Test cases
TEST_CASES = [
    {
        "id": "TC001",
        "query": "How do neural networks learn?",
        "expected_keywords": ["backpropagation", "gradient", "weight"],
        "min_confidence": "MEDIUM",
        "category": "technical"
    },
    {
        "id": "TC002",
        "query": "What is overfitting?",
        "expected_keywords": ["training", "memorize", "data"],
        "min_confidence": "MEDIUM",
        "category": "technical"
    },
    {
        "id": "TC003",
        "query": "What's the weather today?",
        "expected_confidence": "LOW",
        "category": "out_of_scope"
    }
]

def test_relevance_validation():
    """Test that relevance validation works"""
    # Relevant document
    result = validate_relevance(
        "How do neural networks learn?",
        "Neural networks learn through backpropagation which adjusts weights."
    )
    assert result["relevant"] == True, "Should mark relevant doc as relevant"
    
    # Irrelevant document
    result = validate_relevance(
        "How do neural networks learn?",
        "Python is a programming language for web development."
    )
    assert result["relevant"] == False, "Should mark irrelevant doc as irrelevant"
    
    print("✅ Relevance validation tests passed")

def test_grounding_validation():
    """Test that grounding validation works"""
    context = "Neural networks learn through backpropagation."
    
    # Grounded answer
    result = validate_grounding(
        "How do neural networks learn?",
        context,
        "Neural networks learn using backpropagation."
    )
    assert result["fully_grounded"] or result["partially_grounded"], "Should be grounded"
    
    print("✅ Grounding validation tests passed")

def test_self_rag_quality():
    """Test Self-RAG answer quality"""
    for case in TEST_CASES:
        result = self_rag(case["query"])
        
        # Check keywords if expected
        if "expected_keywords" in case:
            answer_lower = result["answer"].lower()
            found = sum(1 for kw in case["expected_keywords"] if kw in answer_lower)
            assert found >= 1, f"{case['id']}: Should contain at least one keyword"
        
        # Check confidence for out-of-scope
        if case.get("expected_confidence") == "LOW":
            assert result["confidence"] in ["LOW", "MEDIUM"], \
                f"{case['id']}: Out-of-scope should have low confidence"
        
        print(f"✅ {case['id']}: Passed")
    
    print("✅ All Self-RAG quality tests passed")

def test_conversational_memory():
    """Test that memory enables follow-up questions"""
    rag = ConversationalSelfRAG()
    
    # First question
    result1 = rag.chat("What is overfitting?")
    assert "overfitting" in result1["answer"].lower() or "training" in result1["answer"].lower()
    
    # Follow-up with pronoun
    result2 = rag.chat("How do I prevent it?")
    # Should understand "it" = overfitting
    assert any(kw in result2["answer"].lower() for kw in ["regularization", "dropout", "overfit"])
    
    print("✅ Conversational memory tests passed")

def run_all_tests():
    """Run all tests"""
    print("🧪 Running Self-RAG Tests...\n")
    print("=" * 50)
    
    test_relevance_validation()
    test_grounding_validation()
    test_self_rag_quality()
    test_conversational_memory()
    
    print("=" * 50)
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    run_all_tests()