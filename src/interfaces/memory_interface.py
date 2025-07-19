"""
Memory interface definition for the AI Agent System.

This module defines the abstract interface that all memory implementations
must follow to ensure consistent behavior across different storage backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..models.data_models import Message


class MemoryInterface(ABC):
    """
    Abstract interface for memory management in the AI Agent System.
    
    This interface defines the contract for storing and retrieving conversation
    history and context information. Implementations can use different storage
    backends (in-memory, file-based, database, etc.).
    """
    
    @abstractmethod
    def store_message(self, message: Message) -> None:
        """
        Store a message in the memory system.
        
        Args:
            message: The message to store
            
        Raises:
            MemoryError: If the message cannot be stored
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current context information.
        
        Returns:
            Dictionary containing context data
            
        Raises:
            MemoryError: If context cannot be retrieved
        """
        pass
    
    @abstractmethod
    def set_context(self, context: Dict[str, Any]) -> None:
        """
        Set the current context information.
        
        Args:
            context: Dictionary containing context data to store
            
        Raises:
            MemoryError: If context cannot be stored
        """
        pass
    
    @abstractmethod
    def clear_memory(self) -> None:
        """
        Clear all stored messages and context.
        
        Raises:
            MemoryError: If memory cannot be cleared
        """
        pass
    
    @abstractmethod
    def get_message_count(self) -> int:
        """
        Get the total number of stored messages.
        
        Returns:
            Number of messages in memory
            
        Raises:
            MemoryError: If count cannot be retrieved
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass