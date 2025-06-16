from vllm_judge.prompt_builder import PromptBuilder


class TestPromptBuilder:
    """Test PromptBuilder functionality."""
    
    def test_build_messages_basic(self):
        """Test basic message building."""
        messages = PromptBuilder.build_messages(
            content="Test response",
            criteria="Test criteria"
        )
        
        assert isinstance(messages, list)
        assert len(messages) >= 1
        assert any("Test response" in str(msg) for msg in messages)
        assert any("Test criteria" in str(msg) for msg in messages)
    
    def test_build_messages_with_input(self):
        """Test message building with input parameter."""
        messages = PromptBuilder.build_messages(
            content="Paris",
            input="What is the capital of France?",
            criteria="accuracy"
        )
        
        assert isinstance(messages, list)
        # Should include the input/question in the messages
        assert any("What is the capital of France?" in str(msg) for msg in messages)
    
    def test_build_messages_comparison(self):
        """Test message building for comparison."""
        messages = PromptBuilder.build_messages(
            content={"a": "Response A", "b": "Response B"},
            criteria="Which is better?"
        )
        
        assert isinstance(messages, list)
        # Should include both responses
        assert any("Response A" in str(msg) for msg in messages)
        assert any("Response B" in str(msg) for msg in messages)
    
    def test_build_messages_with_rubric(self):
        """Test message building with rubric."""
        rubric = "Score from 1-10 based on clarity"
        messages = PromptBuilder.build_messages(
            content="Clear explanation",
            criteria="clarity",
            rubric=rubric
        )
        
        assert isinstance(messages, list)
        assert any(rubric in str(msg) for msg in messages)
    
    def test_build_messages_with_scale(self):
        """Test message building with numeric scale."""
        messages = PromptBuilder.build_messages(
            content="Good response",
            criteria="quality",
            scale=(1, 10)
        )
        
        assert isinstance(messages, list)
        # Should mention the scale somewhere
        message_text = " ".join(str(msg) for msg in messages)
        assert "1" in message_text and "10" in message_text
    
    def test_build_messages_with_examples(self):
        """Test message building with examples."""
        examples = [
            {
                "content": "Example response",
                "decision": "GOOD",
                "reasoning": "This is good because..."
            }
        ]
        
        messages = PromptBuilder.build_messages(
            content="Test response",
            criteria="quality",
            examples=examples
        )
        
        assert isinstance(messages, list)
        # Should include example content
        assert any("Example response" in str(msg) for msg in messages)
    
    def test_build_messages_with_system_prompt(self):
        """Test message building with custom system prompt."""
        system_prompt = "You are a helpful evaluation assistant."
        messages = PromptBuilder.build_messages(
            content="Test response",
            criteria="helpfulness",
            system_prompt=system_prompt
        )
        
        assert isinstance(messages, list)
        # Should have system message
        assert any(msg.get("role") == "system" for msg in messages if isinstance(msg, dict))
        assert any(system_prompt in str(msg) for msg in messages)
    
    def test_build_messages_with_context(self):
        """Test message building with context."""
        context = "This is a conversation about AI safety."
        messages = PromptBuilder.build_messages(
            content="AI should be safe",
            criteria="relevance",
            context=context
        )
        
        assert isinstance(messages, list)
        assert any(context in str(msg) for msg in messages)
    
    def test_format_messages_as_text(self):
        """Test formatting messages as plain text."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        text = PromptBuilder.format_messages_as_text(messages)
        
        assert isinstance(text, str)
        assert "You are helpful." in text
        assert "Hello" in text
        assert "Hi there!" in text