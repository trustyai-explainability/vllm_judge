from vllm_judge.models import EvaluationResult
from typing import List, Dict, Any
import re
import json
import numpy as np

# Llama Guard 3 parser
def parse_llama_guard_3(response: str) -> EvaluationResult:
    """Parse Llama Guard 3's 'safe/unsafe' format."""
    lines = response.strip().split('\n')
    detection = lines[0].lower().strip()
    reasoning = lines[1].strip() if len(lines) > 1 else "No violations detected"
    
    return EvaluationResult(
        decision=detection,
        reasoning=reasoning,
        score=None,
        metadata={"model_type": "llama_guard_3"}
    )

# Granite Guardian 3.2 parser
granite_guardian_3_2_safe_token = "Yes"
granite_guardian_3_2_risky_token = "No"

## adapted from https://github.com/ibm-granite/granite-guardian/blob/main/cookbooks/granite-guardian-3.2/detailed_guide_vllm.ipynb
def parse_granite_guardian_3_2(choices: List[Dict[str, Any]]) -> EvaluationResult:
    model_type = "granite_guardian_3_2"

    # if choices is a string, parse it as a JSON object
    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except json.JSONDecodeError as e:
            return EvaluationResult(
                decision="Failed",
                reasoning=f"JSON parsing error: {str(e)}",
                score=None,
                metadata={"model_type": model_type}
            )
    if not choices:
        return EvaluationResult(
            decision="Failed",
            reasoning="Empty choices list",
            score=None,
            metadata={"model_type": model_type}
        )
    
    label, prob_label = None, None
    choice: Dict[str, Any] = choices[0]

    # get probabilities of safe and risky tokens
    if choice['logprobs'] is not None:
        prob = get_probabilities(choice['logprobs'])
    else:
        prob = None

    # get label from output
    output: str = choice['message']['content'].strip()
    match = re.search(r"^\w+", output, re.MULTILINE)
    if match:
        res = match.group(0).strip()
    else:
        res = "Failed"

    if granite_guardian_3_2_risky_token.lower() == res.lower():
        label = granite_guardian_3_2_risky_token
    elif granite_guardian_3_2_safe_token.lower() == res.lower():
        label = granite_guardian_3_2_safe_token
    else:
        label = res
    
    # get probability of label
    if prob is not None:
        if label == granite_guardian_3_2_risky_token:
            prob_label = prob[1]
        elif label == granite_guardian_3_2_safe_token:
            prob_label = prob[0]

    # get confidence level from output
    confidence_match = re.search(r'<confidence> (.*?) </confidence>', output)
    if confidence_match:
        confidence_level = confidence_match.group(1).strip()
    else:
        confidence_level = "Unknown"

    return EvaluationResult(
        decision=label,
        reasoning=f"Confidence level: {confidence_level}",
        # score=round(prob_label.item(), 3) if prob_label is not None else None, ## original torch
        score=round(prob_label, 3) if prob_label is not None else None,
        metadata={"model_type": model_type}
    )

## removed torch dependency from 
## https://github.com/ibm-granite/granite-guardian/blob/main/cookbooks/granite-guardian-3.2/detailed_guide_vllm.ipynb
def get_probabilities(logprobs: Dict[str, Any]) -> np.ndarray:
    safe_token_prob = 1e-50
    risky_token_prob = 1e-50
    for token_probs in logprobs['content']:
        for token_prob in token_probs['top_logprobs']:
            decoded_token = token_prob['token']
            if decoded_token.strip().lower() == granite_guardian_3_2_safe_token.lower():
                safe_token_prob += np.exp(token_prob['logprob'])
            if decoded_token.strip().lower() == granite_guardian_3_2_risky_token.lower():
                risky_token_prob += np.exp(token_prob['logprob'])

    ### Original implementation using torch
    # probabilities = torch.softmax(
    #     torch.tensor([math.log(safe_token_prob), math.log(risky_token_prob)]), dim=0
    # )
    # return probabilities

    # ### Why not use simple normalization?
    # total = safe_token_prob + risky_token_prob
    # return (safe_token_prob / total, risky_token_prob / total)

    ### Using numpy
    # softmax equivalent
    log_probs = np.array([np.log(safe_token_prob), np.log(risky_token_prob)])
    exp_probs = np.exp(log_probs - np.max(log_probs))  # Subtract max for stability
    probabilities = exp_probs / np.sum(exp_probs)
    return probabilities