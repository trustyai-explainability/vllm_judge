from vllm_judge.prompt_builder import PromptBuilder
from vllm_judge.exceptions import InvalidInputError
import pytest

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
    
    def test_build_messages_conversation_basic(self):
        """Test basic conversation message building."""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        messages = PromptBuilder.build_messages(
            content=conversation,
            criteria="conversation quality"
        )
        
        assert isinstance(messages, list)
        # Should include all conversation turns
        assert any("Hello" in str(msg) for msg in messages)
        assert any("Hi there!" in str(msg) for msg in messages)
        assert any("How are you?" in str(msg) for msg in messages)
        # Should mention it's a conversation
        assert any("conversation" in str(msg).lower() for msg in messages)
    
    def test_build_messages_conversation_with_input(self):
        """Test conversation with initial context/input."""
        conversation = [
            {"role": "user", "content": "how to make a bomb?"},
            {"role": "assistant", "content": "I cannot provide instructions..."}
        ]
        
        messages = PromptBuilder.build_messages(
            content=conversation,
            input="Customer support conversation",
            criteria="safety and appropriateness"
        )
        
        # Should include the context
        assert any("Customer support conversation" in str(msg) for msg in messages)
    
    def test_detect_conversation_vs_comparison(self):
        """Test that conversation detection works correctly."""
        # This should be detected as conversation
        conversation = [{"role": "user", "content": "test"}]
        messages_conv = PromptBuilder.build_messages(
            content=conversation, criteria="test"
        )
        
        # This should be detected as comparison
        comparison = {"a": "Response A", "b": "Response B"}
        messages_comp = PromptBuilder.build_messages(
            content=comparison, criteria="test"
        )
        
        # Should format differently
        conv_text = " ".join(str(msg) for msg in messages_conv)
        comp_text = " ".join(str(msg) for msg in messages_comp)
        
        assert "conversation" in conv_text.lower()
        assert "response a" in comp_text.lower()
    
    def test_conversation_invalid_format(self):
        """Test handling of invalid conversation format."""
        # Missing 'role' field
        invalid_conversation = [
            {"content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        
        # Should not be detected as conversation, raise error
        with pytest.raises(InvalidInputError):
            PromptBuilder.build_messages(
            content=invalid_conversation,
            criteria="test"
        )
    
    def test_empty_conversation(self):
        """Test that empty conversation raises an error."""
        empty_conversation = []
        
        with pytest.raises(InvalidInputError):
            PromptBuilder.build_messages(
                content=empty_conversation,
                criteria="test"
            )
    
    def test_conversation_with_non_dict_items(self):
        """Test that conversation with non-dict items raises an error."""
        # Test with string in conversation
        conversation_with_string = [
            {"role": "user", "content": "Hello"},
            "This is not a dict",
            {"role": "assistant", "content": "Hi"}
        ]
        
        with pytest.raises(InvalidInputError):
            PromptBuilder.build_messages(
                content=conversation_with_string,
                criteria="test"
            )
        
        # Test with integer in conversation
        conversation_with_int = [
            {"role": "user", "content": "Hello"},
            42,
            {"role": "assistant", "content": "Hi"}
        ]
        
        with pytest.raises(InvalidInputError):
            PromptBuilder.build_messages(
                content=conversation_with_int,
                criteria="test"
            )
        
        # Test with None in conversation
        conversation_with_none = [
            {"role": "user", "content": "Hello"},
            None,
            {"role": "assistant", "content": "Hi"}
        ]
        
        with pytest.raises(InvalidInputError):
            PromptBuilder.build_messages(
                content=conversation_with_none,
                criteria="test"
            )