from typing import List, Dict, Union, Optional, Tuple, Any


class PromptBuilder:
    """Builds prompts for evaluation requests."""
    
    @staticmethod
    def build_messages(
        content: Union[str, Dict[str, str]],
        criteria: str,
        input: Optional[str] = None,
        rubric: Union[str, Dict[Union[int, float], str]] = None,
        scale: Optional[Tuple[int, int]] = None,
        examples: List[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Build chat messages for evaluation.
        
        Args:
            content: Single response or dict with 'a' and 'b' for comparison
            criteria: What to evaluate for
            input: Optional input/question/prompt that the response addresses
            rubric: Evaluation guide
            scale: Numeric scale (min, max)
            examples: Few-shot examples
            system_prompt: Custom system message
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            List of chat messages
        """
        # Detect evaluation type
        is_comparison = isinstance(content, dict) and "a" in content and "b" in content
        
        # System message
        if not system_prompt:
            # TODO: Add more detailed system prompts
            system_prompt = "You are an impartial judge and expert evaluator "
            if is_comparison:
                system_prompt+="comparing responses objectively."
            else:
                system_prompt+="providing objective assessments."
        
        # Output format instructions
        system_prompt+="\nYou must respond in JSON format:\n"
        system_prompt+="""{
    "decision": <your judgment - string|boolean>,
    "reasoning": "<concise explanation of your judgment>",
    "score": <numeric score if requested, otherwise null>
}"""
        system_prompt+="\nDo not include any text in your response except for the JSON object."
        
        # Build user message
        user_content = PromptBuilder._build_user_prompt(
            content=content,
            input=input,
            criteria=criteria,
            rubric=rubric,
            scale=scale,
            examples=examples,
            is_comparison=is_comparison,
            context=context,
            **kwargs
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    
    @staticmethod
    def _build_user_prompt(
        content: Union[str, Dict[str, str]],
        criteria: str,
        rubric: Union[str, Dict[Union[int, float], str]],
        scale: Optional[Tuple[int, int]],
        examples: List[Dict[str, Any]],
        is_comparison: bool,
        context: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs
    ) -> str:
        """Build the user message content."""
        parts = []

        # Add input section if provided
        if input:
            parts.append("Given the following input/question:")
            parts.append(f'"{input}"')
            parts.append("")
        
        # Task description
        if is_comparison:
            if input:
                parts.append(f"Compare how well these two responses address the input for: {criteria}")
            else:
                parts.append(f"Compare these two responses based on: {criteria}")
            if context:
                parts.append(f"\nContext: {context}")
            parts.append(f"\nResponse A:\n{content['a']}")
            parts.append(f"\nResponse B:\n{content['b']}")
        else:
            if input:
                parts.append(f"Evaluate how well this content addresses the input for: {criteria}")
            else:
                parts.append(f"Evaluate the following content based on: {criteria}")
            if context:
                parts.append(f"\nContext: {context}")
            parts.append(f"\nContent to evaluate:\n{content}")
        
        parts.append(f"\nYou must return a decision label/class (your judgement) for the `decision` field and a concise explanation for the `reasoning` field.")

        # Add scale and rubric
        if scale:
            parts.append(f"\nIn addition to these, provide a score from {scale[0]} to {scale[1]}")
            
            if isinstance(rubric, dict):
                parts.append("\nScoring guide:")
                # Sort by score in descending order
                sorted_items = sorted(rubric.items(), key=lambda x: float(x[0]), reverse=True)
                for score, description in sorted_items:
                    parts.append(f"- {score}: {description}")
            elif rubric:
                parts.append(f"\nEvaluation guide: {rubric}")
        elif rubric:
            parts.append(f"\nEvaluation guide: {rubric}")
        
        # Add examples if provided
        if examples:
            parts.append("\nExample evaluations:")
            for i, ex in enumerate(examples, 1):
                parts.append(f"\nExample {i}:")
                
                # Handle different example formats
                if "input" in ex:
                    parts.append(f"Input: {ex['input']}")
                if "content" in ex:
                    parts.append(f"Content: {ex['content']}")
                elif "text" in ex:
                    parts.append(f"Text: {ex['text']}")
                
                if "decision" in ex:
                    parts.append(f"Decision: {ex['decision']}")
                if "score" in ex:
                    parts.append(f"Score: {ex['score']}")
                
                if "reasoning" in ex:
                    parts.append(f"Reasoning: {ex['reasoning']}")
        
        # Add any additional instructions
        if kwargs.get("additional_instructions"):
            parts.append(f"\nAdditional instructions: {kwargs['additional_instructions']}")

        # Output format instructions
        parts.append("\nYou must respond in JSON format:")
        parts.append("""{
    "decision": <your judgment - string|boolean>,
    "reasoning": "<concise explanation of your judgment>",
    "score": <numeric score if requested, otherwise null>
}""")
        
        return "\n".join(parts)
    
    @staticmethod
    def format_messages_as_text(messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages as plain text for completion API.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Formatted text prompt
        """
        parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"\nUser: {content}")
            elif role == "assistant":
                parts.append(f"\nAssistant: {content}")
        
        # Add a prompt for the assistant to respond
        parts.append("\nAssistant:")
        
        return "\n".join(parts)