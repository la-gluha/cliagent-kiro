"""
Persistent conversation memory implementation.

This module provides the PersistentConversationMemory class which extends
ConversationMemory with file-based persistence capabilities.
"""

from typing import Any, Dict, List, Optional
import threading
from .conversation_memory import ConversationMemory
from .memory_persistence import MemoryPersistence
from ..models.data_models import Message
from ..exceptions import MemoryError


class PersistentConversationMemory(ConversationMemory):
    """
    Persistent implementation of conversation memory.
    
    This class extends ConversationMemory to add file-based persistence,
    automatically saving and loading conversation data across sessions.
    """
    
    def __init__(self, storage_dir: Optional[str] = None, auto_save: bool = True):
        """
        Initialize the persistent conversation memory.
        
        Args:
            storage_dir: Directory to store memory files (defaults to temp dir)
            auto_save: Whether to automatically save changes to disk
        """
        super().__init__()
        self._persistence = MemoryPersistence(storage_dir)
        self._auto_save = auto_save
        self._save_lock = threading.RLock()
        
        # Load existing data
        self._load_from_storage()
    
    def _load_from_storage(self) -> None:
        """Load existing data from persistent storage."""
        try:
            with self._save_lock:
                # Load messages
                messages = self._persistence.load_messages()
                for message in messages:
                    # Use parent class method to avoid triggering auto-save
                    super().store_message(message)
                
                # Load context
                context = self._persistence.load_context()
                if context:
                    super().set_context(context)
                    
        except Exception as e:
            # If loading fails, start with empty memory but log the error
            # In a production system, you might want to handle this differently
            pass
    
    def _save_to_storage(self) -> None:
        """Save current data to persistent storage."""
        if not self._auto_save:
            return
        
        try:
            with self._save_lock:
                # Save messages
                messages = super().retrieve_history()
                self._persistence.save_messages(messages)
                
                # Save context
                context = super().get_context()
                self._persistence.save_context(context)
                
        except Exception as e:
            # In a production system, you might want to handle save failures
            # differently (e.g., retry, log, notify user)
            raise MemoryError(f"Failed to save to storage: {str(e)}")
    
    def store_message(self, message: Message) -> None:
        """
        Store a message and persist to storage.
        
        Args:
            message: The message to store
            
        Raises:
            MemoryError: If the message cannot be stored
        """
        super().store_message(message)
        self._save_to_storage()
    
    def set_context(self, context: Dict[str, Any]) -> None:
        """
        Set context and persist to storage.
        
        Args:
            context: Dictionary containing context data to store
            
        Raises:
            MemoryError: If context cannot be stored
        """
        super().set_context(context)
        self._save_to_storage()
    
    def update_context(self, updates: Dict[str, Any]) -> None:
        """
        Update context and persist to storage.
        
        Args:
            updates: Dictionary of key-value pairs to update in context
            
        Raises:
            MemoryError: If context cannot be updated
        """
        super().update_context(updates)
        self._save_to_storage()
    
    def remove_context_key(self, key: str) -> bool:
        """
        Remove a context key and persist to storage.
        
        Args:
            key: The key to remove
            
        Returns:
            True if the key was removed, False if it didn't exist
            
        Raises:
            MemoryError: If context cannot be modified
        """
        result = super().remove_context_key(key)
        if result:  # Only save if something was actually removed
            self._save_to_storage()
        return result
    
    def clear_memory(self) -> None:
        """
        Clear all memory and persistent storage.
        
        Raises:
            MemoryError: If memory cannot be cleared
        """
        super().clear_memory()
        try:
            with self._save_lock:
                self._persistence.clear_storage()
        except Exception as e:
            raise MemoryError(f"Failed to clear persistent storage: {str(e)}")
    
    def force_save(self) -> None:
        """
        Force save current state to storage regardless of auto_save setting.
        
        Raises:
            MemoryError: If save fails
        """
        try:
            with self._save_lock:
                messages = super().retrieve_history()
                self._persistence.save_messages(messages)
                
                context = super().get_context()
                self._persistence.save_context(context)
        except Exception as e:
            raise MemoryError(f"Failed to force save: {str(e)}")
    
    def reload_from_storage(self) -> None:
        """
        Reload data from persistent storage, discarding current in-memory state.
        
        Raises:
            MemoryError: If reload fails
        """
        try:
            with self._save_lock:
                # Clear current memory
                super().clear_memory()
                
                # Reload from storage
                self._load_from_storage()
        except Exception as e:
            raise MemoryError(f"Failed to reload from storage: {str(e)}")
    
    def create_backup(self, backup_dir: str) -> None:
        """
        Create a backup of the persistent storage.
        
        Args:
            backup_dir: Directory to store the backup
            
        Raises:
            MemoryError: If backup cannot be created
        """
        try:
            with self._save_lock:
                # Ensure current state is saved before backup
                self.force_save()
                self._persistence.backup_storage(backup_dir)
        except Exception as e:
            if isinstance(e, MemoryError):
                raise
            raise MemoryError(f"Failed to create backup: {str(e)}")
    
    def restore_from_backup(self, backup_dir: str) -> None:
        """
        Restore from a backup and reload memory.
        
        Args:
            backup_dir: Directory containing the backup
            
        Raises:
            MemoryError: If restore fails
        """
        try:
            with self._save_lock:
                self._persistence.restore_from_backup(backup_dir)
                self.reload_from_storage()
        except Exception as e:
            if isinstance(e, MemoryError):
                raise
            raise MemoryError(f"Failed to restore from backup: {str(e)}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about the persistent storage.
        
        Returns:
            Dictionary containing storage information
            
        Raises:
            MemoryError: If storage info cannot be retrieved
        """
        try:
            with self._save_lock:
                info = self._persistence.get_storage_info()
                info["auto_save"] = self._auto_save
                info["in_memory_message_count"] = super().get_message_count()
                return info
        except Exception as e:
            if isinstance(e, MemoryError):
                raise
            raise MemoryError(f"Failed to get storage info: {str(e)}")
    
    def set_auto_save(self, auto_save: bool) -> None:
        """
        Enable or disable automatic saving.
        
        Args:
            auto_save: Whether to automatically save changes
        """
        if not isinstance(auto_save, bool):
            raise MemoryError("auto_save must be a boolean")
        
        with self._save_lock:
            self._auto_save = auto_save
    
    def get_auto_save(self) -> bool:
        """
        Get the current auto-save setting.
        
        Returns:
            Current auto-save setting
        """
        return self._auto_save
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with final save."""
        try:
            # Always save on exit, regardless of auto_save setting
            self.force_save()
        except:
            # Don't raise exceptions during cleanup
            pass