"""Session manager for handling user sessions and state."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class UserSession:
    """Represents a user session with its state and history."""
    session_id: str
    start_time: datetime
    last_activity: datetime
    command_history: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


class SessionManager:
    """Manages user sessions and their state."""
    
    def __init__(self):
        self.current_session: Optional[UserSession] = None
        self.session_history: Dict[str, UserSession] = {}
        self.max_history_size = 1000
        self.max_sessions = 10
    
    def create_session(self) -> UserSession:
        """
        Create a new user session.
        
        Returns:
            New UserSession object
        """
        session_id = str(uuid.uuid4())[:8]  # Short ID for display
        now = datetime.now()
        
        session = UserSession(
            session_id=session_id,
            start_time=now,
            last_activity=now
        )
        
        # Clean up old sessions if we have too many
        if len(self.session_history) >= self.max_sessions:
            oldest_session_id = min(
                self.session_history.keys(),
                key=lambda sid: self.session_history[sid].last_activity
            )
            del self.session_history[oldest_session_id]
        
        self.session_history[session_id] = session
        self.current_session = session
        
        return session
    
    def get_current_session(self) -> Optional[UserSession]:
        """
        Get the current active session.
        
        Returns:
            Current UserSession or None if no active session
        """
        return self.current_session
    
    def update_session_activity(self, command: str) -> None:
        """
        Update the current session's activity.
        
        Args:
            command: The command that was executed
        """
        if not self.current_session:
            self.create_session()
        
        if self.current_session:
            self.current_session.last_activity = datetime.now()
            self.current_session.command_history.append(command)
            
            # Limit history size
            if len(self.current_session.command_history) > self.max_history_size:
                self.current_session.command_history = \
                    self.current_session.command_history[-self.max_history_size:]
    
    def get_command_history(self, limit: Optional[int] = None) -> List[str]:
        """
        Get command history for the current session.
        
        Args:
            limit: Maximum number of commands to return
            
        Returns:
            List of command strings
        """
        if not self.current_session:
            return []
        
        history = self.current_session.command_history
        if limit:
            return history[-limit:]
        return history.copy()
    
    def set_session_context(self, key: str, value: Any) -> None:
        """
        Set a context value for the current session.
        
        Args:
            key: Context key
            value: Context value
        """
        if not self.current_session:
            self.create_session()
        
        if self.current_session:
            self.current_session.context[key] = value
    
    def get_session_context(self, key: str, default: Any = None) -> Any:
        """
        Get a context value from the current session.
        
        Args:
            key: Context key
            default: Default value if key not found
            
        Returns:
            Context value or default
        """
        if not self.current_session:
            return default
        
        return self.current_session.context.get(key, default)
    
    def clear_session_context(self) -> None:
        """Clear all context for the current session."""
        if self.current_session:
            self.current_session.context.clear()
    
    def end_session(self) -> None:
        """End the current session."""
        if self.current_session:
            self.current_session.is_active = False
            self.current_session = None
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about the current session.
        
        Returns:
            Dictionary with session information
        """
        if not self.current_session:
            return {"status": "No active session"}
        
        session = self.current_session
        duration = datetime.now() - session.start_time
        
        return {
            "session_id": session.session_id,
            "start_time": session.start_time.isoformat(),
            "duration": str(duration).split('.')[0],  # Remove microseconds
            "commands_executed": len(session.command_history),
            "last_activity": session.last_activity.isoformat(),
            "context_keys": list(session.context.keys()),
            "is_active": session.is_active
        }
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions.
        
        Returns:
            List of session information dictionaries
        """
        sessions_info = []
        for session in self.session_history.values():
            duration = datetime.now() - session.start_time
            sessions_info.append({
                "session_id": session.session_id,
                "start_time": session.start_time.isoformat(),
                "duration": str(duration).split('.')[0],
                "commands_executed": len(session.command_history),
                "is_active": session.is_active,
                "is_current": session == self.current_session
            })
        
        # Sort by start time, most recent first
        sessions_info.sort(key=lambda x: x["start_time"], reverse=True)
        return sessions_info