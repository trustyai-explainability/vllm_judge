from vllm_judge.models import EvaluationResult

# Llama Guard 3 parser
def parse_llama_guard_3(response: str) -> EvaluationResult:
    """Parse Llama Guard 3's 'safe/unsafe' format."""
    lines = response.strip().split('\n')
    is_safe = lines[0].lower().strip() == 'safe'
    
    return EvaluationResult(
        decision="safe" if is_safe else "unsafe",
        reasoning=lines[1] if len(lines) > 1 else "No violations detected",
        score=None,
        metadata={"model_type": "llama_guard_3"}
    )