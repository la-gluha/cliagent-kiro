"""
Unit tests for the memory system.

This module contains comprehensive tests for the ConversationMemory class
and its implementation of the MemoryInterface.
"""

import pytest
from datetime import datetime, timedelta
import threading
import time
from src.memory.conversation_memory import ConversationMemory
from src.models.data_models import Message, MessageRole
from src.exceptions import MemoryError


class TestConversationMemory:
    """Test cases for the ConversationMemory class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.memory = ConversationMemory()
        
        # Create sample messages for testing
        self.sample_messages = [
            Message(
                content="Hello, how are you?",
                role=MessageRole.USER,
                id="msg1"
            ),
            Message(
                content="I'm doing well, thank you!",
                role=MessageRole.AGENT,
                id="msg2"
            ),
            Message(
                content="System initialization complete",
                role=MessageRole.SYSTEM,
                id="msg3"
            )
        ]
    
    def test_store_message_success(self):
        """Test successful message storage."""
        message = self.sample_messages[0]
        
        # Store the message
        self.memory.store_message(message)
        
        # Verify it was stored
        assert self.memory.get_message_count() == 1
        retrieved = self.memory.retrieve_history()
        assert len(retrieved) == 1
        assert retrieved[0].id == message.id
        assert retrieved[0].content == message.content
    
    def test_store_message_invalid_type(self):
        """Test storing invalid message type."""
        with pytest.raises(MemoryError, match="Invalid message type"):
            self.memory.store_message("not a message")
    
    def test_store_duplicate_message_id(self):
        """Test storing messages with duplicate IDs."""
        message1 = self.sample_messages[0]
        message2 = Message(
            content="Different content",
            role=MessageRole.USER,
            id=message1.id  # Same ID
        )
        
        # Store first message
        self.memory.store_message(message1)
        
        # Attempt to store duplicate ID
        with pytest.raises(MemoryError, match="already exists"):
            self.memory.store_message(message2)
    
    def test_retrieve_history_all(self):
        """Test retrieving all conversation history."""
        # Store multiple messages
        for msg in self.sample_messages:
            self.memory.store_message(msg)
        
        # Retrieve all history
        history = self.memory.retrieve_history()
        
        assert len(history) == len(self.sample_messages)
        for i, msg in enumerate(history):
            assert msg.id == self.sample_messages[i].id
    
    def test_retrieve_history_with_limit(self):
        """Test retrieving history with limit."""
        # Store multiple messages
        for msg in self.sample_messages:
            self.memory.store_message(msg)
        
        # Retrieve with limit
        history = self.memory.retrieve_history(limit=2)
        
        assert len(history) == 2
        # Should get the most recent 2 messages
        assert history[0].id == self.sample_messages[1].id
        assert history[1].id == self.sample_messages[2].id
    
    def test_retrieve_history_limit_zero(self):
        """Test retrieving history with limit of zero."""
        for msg in self.sample_messages:
            self.memory.store_message(msg)
        
        history = self.memory.retrieve_history(limit=0)
        assert len(history) == 0
    
    def test_retrieve_history_invalid_limit(self):
        """Test retrieving history with invalid limit."""
        with pytest.raises(MemoryError, match="non-negative integer"):
            self.memory.retrieve_history(limit=-1)
        
        with pytest.raises(MemoryError, match="non-negative integer"):
            self.memory.retrieve_history(limit="invalid")
    
    def test_context_operations(self):
        """Test context get/set operations."""
        test_context = {
            "user_name": "Alice",
            "session_id": "12345",
            "preferences": {"theme": "dark"}
        }
        
        # Set context
        self.memory.set_context(test_context)
        
        # Get context
        retrieved_context = self.memory.get_context()
        assert retrieved_context == test_context
        
        # Verify it's a copy (modifications don't affect stored context)
        retrieved_context["new_key"] = "new_value"
        original_context = self.memory.get_context()
        assert "new_key" not in original_context
    
    def test_set_context_invalid_type(self):
        """Test setting context with invalid type."""
        with pytest.raises(MemoryError, match="must be a dictionary"):
            self.memory.set_context("not a dict")
    
    def test_clear_memory(self):
        """Test clearing all memory."""
        # Store messages and context
        for msg in self.sample_messages:
            self.memory.store_message(msg)
        
        self.memory.set_context({"key": "value"})
        
        # Verify data exists
        assert self.memory.get_message_count() > 0
        assert len(self.memory.get_context()) > 0
        
        # Clear memory
        self.memory.clear_memory()
        
        # Verify everything is cleared
        assert self.memory.get_message_count() == 0
        assert len(self.memory.get_context()) == 0
        assert len(self.memory.retrieve_history()) == 0
    
    def test_get_message_count(self):
        """Test getting message count."""
        assert self.memory.get_message_count() == 0
        
        # Add messages one by one
        for i, msg in enumerate(self.sample_messages):
            self.memory.store_message(msg)
            assert self.memory.get_message_count() == i + 1
    
    def test_get_messages_by_role(self):
        """Test filtering messages by role."""
        # Store messages with different roles
        for msg in self.sample_messages:
            self.memory.store_message(msg)
        
        # Get user messages
        user_messages = self.memory.get_messages_by_role("user")
        assert len(user_messages) == 1
        assert user_messages[0].role == MessageRole.USER
        
        # Get agent messages
        agent_messages = self.memory.get_messages_by_role("agent")
        assert len(agent_messages) == 1
        assert agent_messages[0].role == MessageRole.AGENT
        
        # Get system messages
        system_messages = self.memory.get_messages_by_role("system")
        assert len(system_messages) == 1
        assert system_messages[0].role == MessageRole.SYSTEM
    
    def test_get_messages_by_invalid_role(self):
        """Test filtering by invalid role."""
        with pytest.raises(MemoryError, match="Invalid role"):
            self.memory.get_messages_by_role("invalid_role")
        
        with pytest.raises(MemoryError, match="must be a string"):
            self.memory.get_messages_by_role(123)
    
    def test_search_messages(self):
        """Test searching messages by content."""
        # Store messages
        for msg in self.sample_messages:
            self.memory.store_message(msg)
        
        # Search for specific content
        results = self.memory.search_messages("Hello")
        assert len(results) == 1
        assert "Hello" in results[0].content
        
        # Case-insensitive search
        results = self.memory.search_messages("hello")
        assert len(results) == 1
        
        # Search with no results
        results = self.memory.search_messages("nonexistent")
        assert len(results) == 0
        
        # Search that matches multiple messages
        self.memory.store_message(Message(
            content="Hello again!",
            role=MessageRole.USER,
            id="msg4"
        ))
        results = self.memory.search_messages("Hello")
        assert len(results) == 2
    
    def test_search_messages_invalid_query(self):
        """Test searching with invalid query."""
        with pytest.raises(MemoryError, match="must be a string"):
            self.memory.search_messages(123)
        
        with pytest.raises(MemoryError, match="cannot be empty"):
            self.memory.search_messages("")
        
        with pytest.raises(MemoryError, match="cannot be empty"):
            self.memory.search_messages("   ")
    
    def test_get_message_by_id(self):
        """Test retrieving message by ID."""
        message = self.sample_messages[0]
        self.memory.store_message(message)
        
        # Retrieve by ID
        retrieved = self.memory.get_message_by_id(message.id)
        assert retrieved is not None
        assert retrieved.id == message.id
        assert retrieved.content == message.content
        
        # Try non-existent ID
        not_found = self.memory.get_message_by_id("nonexistent")
        assert not_found is None
    
    def test_get_message_by_invalid_id(self):
        """Test retrieving message with invalid ID."""
        with pytest.raises(MemoryError, match="must be a string"):
            self.memory.get_message_by_id(123)
    
    def test_get_recent_messages(self):
        """Test getting recent messages."""
        # Store messages
        for msg in self.sample_messages:
            self.memory.store_message(msg)
        
        # Get recent messages
        recent = self.memory.get_recent_messages(2)
        assert len(recent) == 2
        assert recent[0].id == self.sample_messages[1].id
        assert recent[1].id == self.sample_messages[2].id
        
        # Get more than available
        all_recent = self.memory.get_recent_messages(10)
        assert len(all_recent) == 3
        
        # Get zero messages
        none_recent = self.memory.get_recent_messages(0)
        assert len(none_recent) == 0
    
    def test_get_recent_messages_invalid_count(self):
        """Test getting recent messages with invalid count."""
        with pytest.raises(MemoryError, match="non-negative integer"):
            self.memory.get_recent_messages(-1)
        
        with pytest.raises(MemoryError, match="non-negative integer"):
            self.memory.get_recent_messages("invalid")
    
    def test_update_context(self):
        """Test updating context."""
        # Set initial context
        initial_context = {"key1": "value1", "key2": "value2"}
        self.memory.set_context(initial_context)
        
        # Update context
        updates = {"key2": "updated_value", "key3": "new_value"}
        self.memory.update_context(updates)
        
        # Verify updates
        context = self.memory.get_context()
        assert context["key1"] == "value1"  # Unchanged
        assert context["key2"] == "updated_value"  # Updated
        assert context["key3"] == "new_value"  # New
    
    def test_update_context_invalid_type(self):
        """Test updating context with invalid type."""
        with pytest.raises(MemoryError, match="must be a dictionary"):
            self.memory.update_context("not a dict")
    
    def test_remove_context_key(self):
        """Test removing context key."""
        # Set context
        context = {"key1": "value1", "key2": "value2"}
        self.memory.set_context(context)
        
        # Remove existing key
        result = self.memory.remove_context_key("key1")
        assert result is True
        
        updated_context = self.memory.get_context()
        assert "key1" not in updated_context
        assert "key2" in updated_context
        
        # Try to remove non-existent key
        result = self.memory.remove_context_key("nonexistent")
        assert result is False
    
    def test_remove_context_key_invalid_type(self):
        """Test removing context key with invalid type."""
        with pytest.raises(MemoryError, match="must be a string"):
            self.memory.remove_context_key(123)
    
    def test_thread_safety(self):
        """Test thread safety of memory operations."""
        num_threads = 10
        messages_per_thread = 5
        threads = []
        results = []
        
        def store_messages(thread_id):
            """Store messages in a thread."""
            thread_results = []
            for i in range(messages_per_thread):
                try:
                    message = Message(
                        content=f"Thread {thread_id} message {i}",
                        role=MessageRole.USER,
                        id=f"thread_{thread_id}_msg_{i}"
                    )
                    self.memory.store_message(message)
                    thread_results.append(True)
                except Exception as e:
                    thread_results.append(False)
            results.append(thread_results)
        
        # Create and start threads
        for i in range(num_threads):
            thread = threading.Thread(target=store_messages, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all operations succeeded
        total_expected = num_threads * messages_per_thread
        assert self.memory.get_message_count() == total_expected
        
        # Verify all results were successful
        for thread_results in results:
            assert all(thread_results)
    
    def test_message_ordering(self):
        """Test that messages maintain chronological order."""
        messages = []
        
        # Create messages with specific timestamps
        base_time = datetime.now()
        for i in range(5):
            message = Message(
                content=f"Message {i}",
                role=MessageRole.USER,
                id=f"msg_{i}",
                timestamp=base_time + timedelta(seconds=i)
            )
            messages.append(message)
            self.memory.store_message(message)
        
        # Retrieve history and verify order
        history = self.memory.retrieve_history()
        for i, msg in enumerate(history):
            assert msg.id == f"msg_{i}"
            assert msg.content == f"Message {i}"
    
    def test_empty_memory_operations(self):
        """Test operations on empty memory."""
        # Test retrieving from empty memory
        assert self.memory.retrieve_history() == []
        assert self.memory.get_message_count() == 0
        assert self.memory.get_context() == {}
        assert self.memory.get_messages_by_role("user") == []
        assert self.memory.search_messages("anything") == []
        assert self.memory.get_message_by_id("anything") is None
        assert self.memory.get_recent_messages(5) == []
        
        # Test removing from empty context
        assert self.memory.remove_context_key("nonexistent") is False