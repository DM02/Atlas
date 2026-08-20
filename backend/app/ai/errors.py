class AIProviderError(Exception):
    """Raised when an embedding/LLM/reranker provider call fails: rate limit, timeout,
    auth, quota exhaustion, etc. Callers depend on this generic type, not on any
    concrete SDK's exception hierarchy — keeps the provider abstraction real.
    """
