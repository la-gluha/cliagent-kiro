"""Unit tests for the SessionManager class."""

import pytest
from datetime import datetime, timedelta
from src.cli.session_manager import SessionManager, UserSession


class TestSessionManager:
    """Test cases for SessionManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.session_manager = SessionManager()
    
    def test_create_session(self):
        """Test creating a new session."""
        session = self.session_manager.create_session()
        
        assert isinstance(session, UserSession)
        assert session.session_id is not None
        assert len(session.session_id) == 8  # Short ID
        assert session.is_active is True
        assert session.command_history == []
        assert session.context == {}
        assert session == self.session_manager.get_current_session()
    
    def test_get_current_session_none(self):
        """Test getting current session when none exists."""
        assert self.session_manager.get_current_session() is None
    
    def test_update_session_activity(self):
        """Test updating session activity."""
        # Should create session if none exists
        self.session_manager.update_session_activity("test command")
        
        session = self.session_manager.get_current_session()
        assert session is not None
        assert "test command" in session.command_history
        assert session.last_activity is not None
    
    def test_update_session_activity_existing(self):
        """Test updating activity on existing session."""
        session = self.session_manager.create_session()
        original_time = session.last_activity
        
        # Small delay to ensure time difference
        import time
        time.sleep(0.01)
        
        self.session_manager.update_session_activity("new command")
        
        assert session.last_activity > original_time
        assert "new command" in session.command_history
    
    def test_get_command_history_empty(self):
        """Test getting command history when empty."""
        history = self.session_manager.get_command_history()
        assert history == []
    
    def test_get_command_history_with_commands(self):
        """Test getting command history with commands."""
        commands = ["help", "task do something", "exit"]
        
        for cmd in commands:
            self.session_manager.update_session_activity(cmd)
        
        history = self.session_manager.get_command_history()
        assert history == commands
    
    def test_get_command_history_with_limit(self):
        """Test getting limited command history."""
        commands = ["cmd1", "cmd2", "cmd3", "cmd4", "cmd5"]
        
        for cmd in commands:
            self.session_manager.update_session_activity(cmd)
        
        history = self.session_manager.get_command_history(limit=3)
        assert history == ["cmd3", "cmd4", "cmd5"]
    
    def test_command_history_size_limit(self):
        """Test that command history respects size limit."""
        # Set a small limit for testing
        self.session_manager.max_history_size = 3
        
        commands = ["cmd1", "cmd2", "cmd3", "cmd4", "cmd5"]
        for cmd in commands:
            self.session_manager.update_session_activity(cmd)
        
        history = self.session_manager.get_command_history()
        assert len(history) == 3
        assert history == ["cmd3", "cmd4", "cmd5"]
    
    def test_set_get_session_context(self):
        """Test setting and getting session context."""
        self.session_manager.create_session()
        
        self.session_manager.set_session_context("key1", "value1")
        self.session_manager.set_session_context("key2", 42)
        
        assert self.session_manager.get_session_context("key1") == "value1"
        assert self.session_manager.get_session_context("key2") == 42
        assert self.session_manager.get_session_context("nonexistent") is None
        assert self.session_manager.get_session_context("nonexistent", "default") == "default"
    
    def test_set_context_creates_session(self):
        """Test that setting context creates session if none exists."""
        assert self.session_manager.get_current_session() is None
        
        self.session_manager.set_session_context("key", "value")
        
        session = self.session_manager.get_current_session()
        assert session is not None
        assert session.context["key"] == "value"
    
    def test_clear_session_context(self):
        """Test clearing session context."""
        self.session_manager.create_session()
        self.session_manager.set_session_context("key1", "value1")
        self.session_manager.set_session_context("key2", "value2")
        
        self.session_manager.clear_session_context()
        
        session = self.session_manager.get_current_session()
        assert session.context == {}
    
    def test_end_session(self):
        """Test ending a session."""
        session = self.session_manager.create_session()
        session_id = session.session_id
        
        self.session_manager.end_session()
        
        assert self.session_manager.get_current_session() is None
        assert not self.session_manager.session_history[session_id].is_active
    
    def test_get_session_info(self):
        """Test getting session information."""
        # Test with no session
        info = self.session_manager.get_session_info()
        assert info["status"] == "No active session"
        
        # Test with active session
        session = self.session_manager.create_session()
        self.session_manager.update_session_activity("test command")
        self.session_manager.set_session_context("test_key", "test_value")
        
        info = self.session_manager.get_session_info()
        
        assert info["session_id"] == session.session_id
        assert "start_time" in info
        assert "duration" in info
        assert info["commands_executed"] == 1
        assert "last_activity" in info
        assert "test_key" in info["context_keys"]
        assert info["is_active"] is True
    
    def test_list_sessions_empty(self):
        """Test listing sessions when none exist."""
        sessions = self.session_manager.list_sessions()
        assert sessions == []
    
    def test_list_sessions_with_data(self):
        """Test listing sessions with data."""
        # Create multiple sessions
        session1 = self.session_manager.create_session()
        self.session_manager.update_session_activity("command1")
        self.session_manager.end_session()
        
        # Small delay to ensure different timestamps
        import time
        time.sleep(0.01)
        
        session2 = self.session_manager.create_session()
        self.session_manager.update_session_activity("command2")
        
        sessions = self.session_manager.list_sessions()
        
        assert len(sessions) == 2
        
        # Should be sorted by start time, most recent first
        assert sessions[0]["session_id"] == session2.session_id
        assert sessions[0]["is_current"] is True
        assert sessions[0]["is_active"] is True
        
        assert sessions[1]["session_id"] == session1.session_id
        assert sessions[1]["is_current"] is False
        assert sessions[1]["is_active"] is False
    
    def test_max_sessions_limit(self):
        """Test that old sessions are cleaned up when limit is reached."""
        # Set a small limit for testing
        self.session_manager.max_sessions = 2
        
        # Create sessions beyond the limit
        session1 = self.session_manager.create_session()
        session1_id = session1.session_id
        self.session_manager.end_session()
        
        session2 = self.session_manager.create_session()
        session2_id = session2.session_id
        self.session_manager.end_session()
        
        session3 = self.session_manager.create_session()
        session3_id = session3.session_id
        
        # Should only have 2 sessions, oldest should be removed
        assert len(self.session_manager.session_history) == 2
        assert session1_id not in self.session_manager.session_history
        assert session2_id in self.session_manager.session_history
        assert session3_id in self.session_manager.session_history
    
    def test_session_timestamps(self):
        """Test that session timestamps are properly maintained."""
        session = self.session_manager.create_session()
        start_time = session.start_time
        
        # Small delay
        import time
        time.sleep(0.01)
        
        self.session_manager.update_session_activity("test")
        
        assert session.last_activity > start_time
        assert session.start_time == start_time  # Should not change