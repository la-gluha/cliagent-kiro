"""
In-memory conversation storage implementation.

This module provides the ConversationMemory class which implements the
MemoryInterface for storing conversation history and context in memory.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import threading
from ..interfaces.memory_interface import MemoryInterface
from ..models.data_models import Message, MessageRole
from ..exceptions import MemoryError


class ConversationMemory(MemoryInterface):
    """
    In-memory implementation of the MemoryInterface.
    
    This class stores conversation history and context in memory with
    thread-safe operations and proper indexing for efficient retrieval.
    """
    
    def __init__(self):
        """Initialize the conversation memory."""
        self._messages: List[Message] = []
        self._context: Dict[str, Any] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._message_index: Dict[str, int] = {}  # Message ID to index mapping
    
    def store_message(self, message: Message) -> None:
        """
        Store a message in memory.
        
        Args:
            message: The message to store
            
        Raises:
            MemoryError: If the message cannot be stored
        """
        if not isinstance(message, Message):
            raise MemoryError("Invalid message type: expected Message instance")
        
        try:
            with self._lock:
                # Validate the message
                message.validate()
                
                # Check for duplicate message IDs
                if message.id in self._message_index:
                    raise MemoryError(f"Message with ID {message.id} already exists")
                
                # Store the message
                index = len(self._messages)
                self._messages.append(message)
                self._message_index[message.id] = index
                
        except Exception as e:
            if isinstance(e, MemoryError):
                raise
            raise MemoryError(f"Failed to store message: {str(e)}")
    
    def retrieve_history(self, limit: Optional[int] = None) -> List[Message]:
        """
        Retrieve conversation history from memory.
        
        Args:
            limit: Maximum number of messages to retrieve (None for all)
            
        Returns:
            List of messages in chronological order (oldest first)
            
        Raises:
            MemoryError: If history cannot be retrieved
        """
        try:
            with self._lock:
                if limit is None:
                    return self._messages.copy()
                
                if not isinstance(limit, int) or limit < 0:
                    raise MemoryError("Limit must be a non-negative integer")
                
                if limit == 0:
                    return []
                
                # Return the most recent messages up to the limit
                start_index = max(0, len(self._messages) - limit)
                return self._messages[start_index:].copy()
                
        except Exception as e:
            if isinstance(e, MemoryError):
                raise
            raise MemoryError(f"Failed to retrieve history: {str(e)}")
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current context information.
        
        Returns:
            Dictionary containing context data
            
        Raises:
            MemoryError: If context cannot be retrieved
        """
        try:
            with self._lock:
                return self._context.copy()
        except Exception as e:
            raise MemoryError(f"Failed to retrieve context: {str(e)}")
    
    def set_context(self, context: Dict[str, Any]) -> None:
        """
        Set the current context information.
        
        Args:
            context: Dictionary containing context data to store
            
        Raises:
            MemoryError: If context cannot be stored
        """
        if not isinstance(context, dict):
            raise MemoryError("Context must be a dictionary")
        
        try:
            with self._lock:
                self._context = context.copy()
        except Exception as e:
            raise MemoryError(f"Failed to set context: {str(e)}")
    
    def clear_memory(self) -> None:
        """
        Clear all stored messages and context.
        
        Raises:
            MemoryError: If memory cannot be cleared
        """
        try:
            with self._lock:
                self._messages.clear()
                self._context.clear()
                self._message_index.clear()
        except Exception as e:
            raise MemoryError(f"Failed to clear memory: {str(e)}")
    
    def get_message_count(self) -> int:
        """
        Get the total number of stored messages.
        
        Returns:
            Number of messages in memory
            
        Raises:
            MemoryError: If count cannot be retrieved
        """
        try:
            with self._lock:
                return len(self._messages)
        except Exception as e:
            raise MemoryError(f"Failed to get message count: {str(e)}")
    
    def get_messages_by_role(self, role: str) -> List[Message]:
        """
        Retrieve messages filtered by role.
        
        Args:
            role: The role to filter by (user, agent, system)
            
        Returns:
            List of messages with the specified role
            
        Raises:
            MemoryError: If messages cannot be retrieved
        """
        try:
            # Validate the role
            if isinstance(role, str):
                try:
                    role_enum = MessageRole(role)
                except ValueError:
                    raise MemoryError(f"Invalid role: {role}")
            else:
                raise MemoryError("Role must be a string")
            
            with self._lock:
                filtered_messages = [
                    msg for msg in self._messages 
                    if msg.role == role_enum
                ]
                return filtered_messages
                
        except Exception as e:
            if isinstance(e, MemoryError):
                raise
            raise MemoryError(f"Failed to retrieve messages by role: {str(e)}")
    
    def search_messages(self, query: str) -> List[Message]:
        """
        Search for messages containing the specified query.
        
        Args:
            query: Text to search for in message content
            
        Returns:
            List of messages containing the query
            
        Raises:
            MemoryError: If search cannot be performed
        """
        if not isinstance(query, str):
            raise MemoryError("Query must be a string")
        
        if not query.strip():
            raise MemoryError("Query cannot be empty")
        
        try:
            with self._lock:
                query_lower = query.lower()
                matching_messages = [
                    msg for msg in self._messages
                    if query_lower in msg.content.lower()
                ]
                return matching_messages
                
        except Exception as e:
            if isinstance(e, MemoryError):
                raise
            raise MemoryError(f"Failed to search messages: {str(e)}")
    
    def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """
        Retrieve a specific message by its ID.
        
        Args:
            message_id: The ID of the message to retrieve
            
        Returns:
            The message if found, None otherwise
            
        Raises:
            MemoryError: If retrieval fails
        """
        if not isinstance(message_id, str):
            raise MemoryError("Message ID must be a string")
        
        try:
            with self._lock:
                index = self._message_index.get(message_id)
                if index is not None and 0 <= index < len(self._messages):
                    return self._messages[index]
                return None
                
        except Exception as e:
            raise MemoryError(f"Failed to retrieve message by ID: {str(e)}")
    
    def get_recent_messages(self, count: int) -> List[Message]:
        """
        Get the most recent messages.
        
        Args:
            count: Number of recent messages to retrieve
            
        Returns:
            List of the most recent messages
            
        Raises:
            MemoryError: If messages cannot be retrieved
        """
        if not isinstance(count, int) or count < 0:
            raise MemoryError("Count must be a non-negative integer")
        
        try:
            with self._lock:
                if count == 0:
                    return []
                
                start_index = max(0, len(self._messages) - count)
                return self._messages[start_index:].copy()
                
        except Exception as e:
            raise MemoryError(f"Failed to retrieve recent messages: {str(e)}")
    
    def update_context(self, updates: Dict[str, Any]) -> None:
        """
        Update specific keys in the context.
        
        Args:
            updates: Dictionary of key-value pairs to update in context
            
        Raises:
            MemoryError: If context cannot be updated
        """
        if not isinstance(updates, dict):
            raise MemoryError("Updates must be a dictionary")
        
        try:
            with self._lock:
                self._context.update(updates)
        except Exception as e:
            raise MemoryError(f"Failed to update context: {str(e)}")
    
    def remove_context_key(self, key: str) -> bool:
        """
        Remove a specific key from the context.
        
        Args:
            key: The key to remove
            
        Returns:
            True if the key was removed, False if it didn't exist
            
        Raises:
            MemoryError: If context cannot be modified
        """
        if not isinstance(key, str):
            raise MemoryError("Key must be a string")
        
        try:
            with self._lock:
                if key in self._context:
                    del self._context[key]
                    return True
                return False
        except Exception as e:
            raise MemoryError(f"Failed to remove context key: {str(e)}")