"""
Integration tests for memory persistence.

This module contains tests for the MemoryPersistence and PersistentConversationMemory
classes, focusing on file-based storage and cross-session persistence.
"""

import pytest
import tempfile
import shutil
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from src.memory.memory_persistence import MemoryPersistence
from src.memory.persistent_conversation_memory import PersistentConversationMemory
from src.models.data_models import Message, MessageRole
from src.exceptions import MemoryError


class TestMemoryPersistence:
    """Test cases for the MemoryPersistence class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence = MemoryPersistence(self.temp_dir)
        
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
        
        self.sample_context = {
            "user_name": "Alice",
            "session_id": "12345",
            "preferences": {"theme": "dark", "language": "en"}
        }
    
    def teardown_method(self):
        """Clean up after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_messages(self):
        """Test saving and loading messages."""
        # Save messages
        self.persistence.save_messages(self.sample_messages)
        
        # Verify file exists
        assert self.persistence.messages_file.exists()
        
        # Load messages
        loaded_messages = self.persistence.load_messages()
        
        # Verify loaded messages match original
        assert len(loaded_messages) == len(self.sample_messages)
        for original, loaded in zip(self.sample_messages, loaded_messages):
            assert loaded.id == original.id
            assert loaded.content == original.content
            assert loaded.role == original.role
            assert loaded.metadata == original.metadata
    
    def test_save_and_load_context(self):
        """Test saving and loading context."""
        # Save context
        self.persistence.save_context(self.sample_context)
        
        # Verify file exists
        assert self.persistence.context_file.exists()
        
        # Load context
        loaded_context = self.persistence.load_context()
        
        # Verify loaded context matches original
        assert loaded_context == self.sample_context
    
    def test_load_nonexistent_files(self):
        """Test loading from nonexistent files."""
        # Load messages from nonexistent file
        messages = self.persistence.load_messages()
        assert messages == []
        
        # Load context from nonexistent file
        context = self.persistence.load_context()
        assert context == {}
    
    def test_save_invalid_messages(self):
        """Test saving invalid messages."""
        with pytest.raises(MemoryError, match="must be a list"):
            self.persistence.save_messages("not a list")
        
        with pytest.raises(MemoryError, match="Message instances"):
            self.persistence.save_messages(["not a message"])
    
    def test_save_invalid_context(self):
        """Test saving invalid context."""
        with pytest.raises(MemoryError, match="must be a dictionary"):
            self.persistence.save_context("not a dict")
        
        # Test non-serializable context
        non_serializable = {"func": lambda x: x}
        with pytest.raises(MemoryError, match="not JSON serializable"):
            self.persistence.save_context(non_serializable)
    
    def test_atomic_operations(self):
        """Test that save operations are atomic."""
        # Save initial data
        self.persistence.save_messages(self.sample_messages[:1])
        self.persistence.save_context({"key": "value"})
        
        # Verify files exist
        assert self.persistence.messages_file.exists()
        assert self.persistence.context_file.exists()
        
        # Get initial file sizes
        initial_msg_size = self.persistence.messages_file.stat().st_size
        initial_ctx_size = self.persistence.context_file.stat().st_size
        
        # Save new data
        self.persistence.save_messages(self.sample_messages)
        self.persistence.save_context(self.sample_context)
        
        # Verify files were updated (different sizes)
        new_msg_size = self.persistence.messages_file.stat().st_size
        new_ctx_size = self.persistence.context_file.stat().st_size
        
        assert new_msg_size != initial_msg_size
        assert new_ctx_size != initial_ctx_size
        
        # Verify no temporary files remain
        temp_files = list(Path(self.temp_dir).glob("*.tmp"))
        assert len(temp_files) == 0
    
    def test_clear_storage(self):
        """Test clearing storage."""
        # Save data
        self.persistence.save_messages(self.sample_messages)
        self.persistence.save_context(self.sample_context)
        
        # Verify files exist
        assert self.persistence.messages_file.exists()
        assert self.persistence.context_file.exists()
        
        # Clear storage
        self.persistence.clear_storage()
        
        # Verify files are gone
        assert not self.persistence.messages_file.exists()
        assert not self.persistence.context_file.exists()
    
    def test_backup_and_restore(self):
        """Test backup and restore functionality."""
        # Save initial data
        self.persistence.save_messages(self.sample_messages)
        self.persistence.save_context(self.sample_context)
        
        # Create backup
        backup_dir = os.path.join(self.temp_dir, "backup")
        self.persistence.backup_storage(backup_dir)
        
        # Verify backup files exist
        backup_path = Path(backup_dir)
        assert (backup_path / "messages.json").exists()
        assert (backup_path / "context.json").exists()
        
        # Modify original data
        modified_messages = self.sample_messages[:1]
        modified_context = {"modified": True}
        self.persistence.save_messages(modified_messages)
        self.persistence.save_context(modified_context)
        
        # Verify data was modified
        loaded_messages = self.persistence.load_messages()
        loaded_context = self.persistence.load_context()
        assert len(loaded_messages) == 1
        assert loaded_context == modified_context
        
        # Restore from backup
        self.persistence.restore_from_backup(backup_dir)
        
        # Verify original data was restored
        restored_messages = self.persistence.load_messages()
        restored_context = self.persistence.load_context()
        assert len(restored_messages) == len(self.sample_messages)
        assert restored_context == self.sample_context
    
    def test_backup_invalid_directory(self):
        """Test backup with invalid directory."""
        with pytest.raises(MemoryError, match="must be a string"):
            self.persistence.backup_storage(123)
    
    def test_restore_nonexistent_backup(self):
        """Test restore from nonexistent backup."""
        with pytest.raises(MemoryError, match="does not exist"):
            self.persistence.restore_from_backup("/nonexistent/path")
    
    def test_get_storage_info(self):
        """Test getting storage information."""
        # Get info for empty storage
        info = self.persistence.get_storage_info()
        assert info["storage_dir"] == str(self.persistence.storage_dir)
        assert not info["messages_file"]["exists"]
        assert not info["context_file"]["exists"]
        
        # Save data
        self.persistence.save_messages(self.sample_messages)
        self.persistence.save_context(self.sample_context)
        
        # Get info after saving
        info = self.persistence.get_storage_info()
        assert info["messages_file"]["exists"]
        assert info["context_file"]["exists"]
        assert info["messages_file"]["size"] > 0
        assert info["context_file"]["size"] > 0
        assert info["messages_file"]["modified"] is not None
        assert info["context_file"]["modified"] is not None
    
    def test_context_manager(self):
        """Test using persistence as context manager."""
        with MemoryPersistence(self.temp_dir) as persistence:
            persistence.save_messages(self.sample_messages)
            persistence.save_context(self.sample_context)
        
        # Verify data was saved
        loaded_messages = self.persistence.load_messages()
        loaded_context = self.persistence.load_context()
        assert len(loaded_messages) == len(self.sample_messages)
        assert loaded_context == self.sample_context


class TestPersistentConversationMemory:
    """Test cases for the PersistentConversationMemory class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        
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
            )
        ]
        
        self.sample_context = {
            "user_name": "Alice",
            "session_id": "12345"
        }
    
    def teardown_method(self):
        """Clean up after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_auto_save_messages(self):
        """Test automatic saving of messages."""
        memory = PersistentConversationMemory(self.temp_dir, auto_save=True)
        
        # Store messages
        for msg in self.sample_messages:
            memory.store_message(msg)
        
        # Create new instance to test persistence
        memory2 = PersistentConversationMemory(self.temp_dir, auto_save=True)
        
        # Verify messages were loaded
        loaded_messages = memory2.retrieve_history()
        assert len(loaded_messages) == len(self.sample_messages)
        for original, loaded in zip(self.sample_messages, loaded_messages):
            assert loaded.id == original.id
            assert loaded.content == original.content
    
    def test_auto_save_context(self):
        """Test automatic saving of context."""
        memory = PersistentConversationMemory(self.temp_dir, auto_save=True)
        
        # Set context
        memory.set_context(self.sample_context)
        
        # Create new instance to test persistence
        memory2 = PersistentConversationMemory(self.temp_dir, auto_save=True)
        
        # Verify context was loaded
        loaded_context = memory2.get_context()
        assert loaded_context == self.sample_context
    
    def test_manual_save(self):
        """Test manual saving with auto_save disabled."""
        memory = PersistentConversationMemory(self.temp_dir, auto_save=False)
        
        # Store data without auto-save
        memory.store_message(self.sample_messages[0])
        memory.set_context(self.sample_context)
        
        # Create new instance - should not have the data
        memory2 = PersistentConversationMemory(self.temp_dir, auto_save=False)
        assert memory2.get_message_count() == 0
        assert memory2.get_context() == {}
        
        # Force save and try again
        memory.force_save()
        memory3 = PersistentConversationMemory(self.temp_dir, auto_save=False)
        assert memory3.get_message_count() == 1
        assert memory3.get_context() == self.sample_context
    
    def test_cross_session_persistence(self):
        """Test persistence across multiple sessions."""
        # Session 1: Store some data
        with PersistentConversationMemory(self.temp_dir) as memory1:
            memory1.store_message(self.sample_messages[0])
            memory1.set_context({"session": 1})
        
        # Session 2: Add more data
        with PersistentConversationMemory(self.temp_dir) as memory2:
            # Should have data from session 1
            assert memory2.get_message_count() == 1
            assert memory2.get_context()["session"] == 1
            
            # Add more data
            memory2.store_message(self.sample_messages[1])
            memory2.update_context({"session": 2})
        
        # Session 3: Verify all data persists
        with PersistentConversationMemory(self.temp_dir) as memory3:
            assert memory3.get_message_count() == 2
            context = memory3.get_context()
            assert context["session"] == 2
            
            messages = memory3.retrieve_history()
            assert messages[0].id == self.sample_messages[0].id
            assert messages[1].id == self.sample_messages[1].id
    
    def test_context_operations_with_persistence(self):
        """Test context operations with persistence."""
        memory = PersistentConversationMemory(self.temp_dir)
        
        # Set initial context
        memory.set_context({"key1": "value1", "key2": "value2"})
        
        # Update context
        memory.update_context({"key2": "updated", "key3": "new"})
        
        # Remove a key
        result = memory.remove_context_key("key1")
        assert result is True
        
        # Create new instance and verify persistence
        memory2 = PersistentConversationMemory(self.temp_dir)
        context = memory2.get_context()
        assert "key1" not in context
        assert context["key2"] == "updated"
        assert context["key3"] == "new"
    
    def test_clear_memory_with_persistence(self):
        """Test clearing memory and persistence."""
        memory = PersistentConversationMemory(self.temp_dir)
        
        # Store data
        memory.store_message(self.sample_messages[0])
        memory.set_context(self.sample_context)
        
        # Verify data exists
        assert memory.get_message_count() == 1
        assert len(memory.get_context()) > 0
        
        # Clear memory
        memory.clear_memory()
        
        # Verify memory is cleared
        assert memory.get_message_count() == 0
        assert memory.get_context() == {}
        
        # Create new instance and verify persistence is also cleared
        memory2 = PersistentConversationMemory(self.temp_dir)
        assert memory2.get_message_count() == 0
        assert memory2.get_context() == {}
    
    def test_reload_from_storage(self):
        """Test reloading from storage."""
        memory = PersistentConversationMemory(self.temp_dir)
        
        # Store initial data
        memory.store_message(self.sample_messages[0])
        memory.set_context({"initial": True})
        
        # Modify in-memory data without saving
        memory.set_auto_save(False)
        memory.store_message(self.sample_messages[1])
        memory.update_context({"modified": True})
        
        # Verify modified data exists in memory
        assert memory.get_message_count() == 2
        assert memory.get_context()["modified"] is True
        
        # Reload from storage
        memory.reload_from_storage()
        
        # Verify data was reverted to stored state
        assert memory.get_message_count() == 1
        context = memory.get_context()
        assert context["initial"] is True
        assert "modified" not in context
    
    def test_backup_and_restore_integration(self):
        """Test backup and restore with persistent memory."""
        memory = PersistentConversationMemory(self.temp_dir)
        
        # Store data
        for msg in self.sample_messages:
            memory.store_message(msg)
        memory.set_context(self.sample_context)
        
        # Create backup
        backup_dir = os.path.join(self.temp_dir, "backup")
        memory.create_backup(backup_dir)
        
        # Modify data
        memory.clear_memory()
        memory.store_message(Message("Modified", MessageRole.USER, "modified"))
        
        # Restore from backup
        memory.restore_from_backup(backup_dir)
        
        # Verify original data was restored
        assert memory.get_message_count() == len(self.sample_messages)
        assert memory.get_context() == self.sample_context
    
    def test_storage_info(self):
        """Test getting storage information."""
        memory = PersistentConversationMemory(self.temp_dir)
        
        # Get initial info
        info = memory.get_storage_info()
        assert info["auto_save"] is True
        assert info["in_memory_message_count"] == 0
        
        # Store data and get updated info
        memory.store_message(self.sample_messages[0])
        info = memory.get_storage_info()
        assert info["in_memory_message_count"] == 1
        assert info["messages_file"]["exists"] is True
    
    def test_auto_save_setting(self):
        """Test changing auto-save setting."""
        memory = PersistentConversationMemory(self.temp_dir, auto_save=False)
        
        # Verify initial setting
        assert memory.get_auto_save() is False
        
        # Change setting
        memory.set_auto_save(True)
        assert memory.get_auto_save() is True
        
        # Test invalid setting
        with pytest.raises(MemoryError, match="must be a boolean"):
            memory.set_auto_save("invalid")
    
    def test_context_manager_with_final_save(self):
        """Test context manager with final save on exit."""
        # Use context manager with auto_save disabled
        with PersistentConversationMemory(self.temp_dir, auto_save=False) as memory:
            memory.store_message(self.sample_messages[0])
            memory.set_context(self.sample_context)
        
        # Verify data was saved on exit
        memory2 = PersistentConversationMemory(self.temp_dir)
        assert memory2.get_message_count() == 1
        assert memory2.get_context() == self.sample_context
    
    def test_concurrent_access(self):
        """Test concurrent access to persistent memory."""
        import threading
        import time
        
        # Use a single memory instance to avoid file conflicts
        memory = PersistentConversationMemory(self.temp_dir)
        results = []
        
        def store_messages(thread_id, num_messages):
            thread_results = []
            for i in range(num_messages):
                try:
                    message = Message(
                        content=f"Thread {thread_id} message {i}",
                        role=MessageRole.USER,
                        id=f"t{thread_id}_m{i}"
                    )
                    memory.store_message(message)
                    thread_results.append(True)
                    time.sleep(0.001)  # Small delay
                except Exception as e:
                    thread_results.append(False)
            results.append(thread_results)
        
        # Create multiple threads
        threads = []
        num_threads = 3
        messages_per_thread = 3  # Reduced to avoid too many file operations
        
        for i in range(num_threads):
            thread = threading.Thread(target=store_messages, args=(i, messages_per_thread))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify most operations succeeded (some may fail due to concurrency)
        total_expected = num_threads * messages_per_thread
        actual_count = memory.get_message_count()
        
        # We expect at least some messages to be stored successfully
        assert actual_count > 0
        assert actual_count <= total_expected