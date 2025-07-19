"""
File-based persistence for conversation memory.

This module provides the MemoryPersistence class which handles saving
and loading conversation data to/from JSON files with proper error handling
and context management.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import threading
import tempfile
import shutil
from ..models.data_models import Message
from ..exceptions import MemoryError


class MemoryPersistence:
    """
    File-based persistence manager for conversation memory.
    
    This class handles saving and loading conversation history and context
    to/from JSON files with atomic operations and proper error handling.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the memory persistence manager.
        
        Args:
            storage_dir: Directory to store memory files (defaults to user's temp dir)
        """
        if storage_dir is None:
            storage_dir = os.path.join(tempfile.gettempdir(), "ai_agent_memory")
        
        self.storage_dir = Path(storage_dir)
        self.messages_file = self.storage_dir / "messages.json"
        self.context_file = self.storage_dir / "context.json"
        self._lock = threading.RLock()
        
        # Ensure storage directory exists
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self) -> None:
        """Ensure the storage directory exists."""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise MemoryError(f"Failed to create storage directory: {str(e)}")
    
    def save_messages(self, messages: List[Message]) -> None:
        """
        Save messages to persistent storage.
        
        Args:
            messages: List of messages to save
            
        Raises:
            MemoryError: If messages cannot be saved
        """
        if not isinstance(messages, list):
            raise MemoryError("Messages must be a list")
        
        try:
            with self._lock:
                # Convert messages to serializable format
                serializable_messages = []
                for msg in messages:
                    if not isinstance(msg, Message):
                        raise MemoryError("All items must be Message instances")
                    serializable_messages.append(msg.to_dict())
                
                # Write to temporary file first (atomic operation)
                import uuid
                temp_file = self.messages_file.with_suffix(f'.tmp.{uuid.uuid4().hex[:8]}')
                try:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(serializable_messages, f, indent=2, ensure_ascii=False)
                    
                    # Atomically replace the original file
                    if os.name == 'nt':  # Windows
                        # On Windows, we need to remove the target file first
                        if self.messages_file.exists():
                            self.messages_file.unlink()
                        shutil.move(str(temp_file), str(self.messages_file))
                    else:
                        # On Unix-like systems, move is atomic
                        shutil.move(str(temp_file), str(self.messages_file))
                except Exception:
                    # Clean up temp file on any error
                    if temp_file.exists():
                        try:
                            temp_file.unlink()
                        except:
                            pass
                    raise
                
        except Exception as e:
            if isinstance(e, MemoryError):
                raise
            raise MemoryError(f"Failed to save messages: {str(e)}")
    
    def load_messages(self) -> List[Message]:
        """
        Load messages from persistent storage.
        
        Returns:
            List of messages loaded from storage
            
        Raises:
            MemoryError: If messages cannot be loaded
        """
        try:
            with self._lock:
                if not self.messages_file.exists():
                    return []
                
                with open(self.messages_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    raise MemoryError("Invalid messages file format: expected list")
                
                messages = []
                for item in data:
                    if not isinstance(item, dict):
                        raise MemoryError("Invalid message format: expected dictionary")
                    
                    try:
                        message = Message.from_dict(item)
                        messages.append(message)
                    except Exception as e:
                        raise MemoryError(f"Failed to deserialize message: {str(e)}")
                
                return messages
                
        except Exception as e:
            if isinstance(e, MemoryError):
                raise
            raise MemoryError(f"Failed to load messages: {str(e)}")
    
    def save_context(self, context: Dict[str, Any]) -> None:
        """
        Save context to persistent storage.
        
        Args:
            context: Context dictionary to save
            
        Raises:
            MemoryError: If context cannot be saved
        """
        if not isinstance(context, dict):
            raise MemoryError("Context must be a dictionary")
        
        try:
            with self._lock:
                # Ensure context is JSON serializable
                try:
                    json.dumps(context)
                except (TypeError, ValueError) as e:
                    raise MemoryError(f"Context is not JSON serializable: {str(e)}")
                
                # Write to temporary file first (atomic operation)
                import uuid
                temp_file = self.context_file.with_suffix(f'.tmp.{uuid.uuid4().hex[:8]}')
                try:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(context, f, indent=2, ensure_ascii=False)
                    
                    # Atomically replace the original file
                    if os.name == 'nt':  # Windows
                        # On Windows, we need to remove the target file first
                        if self.context_file.exists():
                            self.context_file.unlink()
                        shutil.move(str(temp_file), str(self.context_file))
                    else:
                        # On Unix-like systems, move is atomic
                        shutil.move(str(temp_file), str(self.context_file))
                except Exception:
                    # Clean up temp file on any error
                    if temp_file.exists():
                        try:
                            temp_file.unlink()
                        except:
                            pass
                    raise
                
        except Exception as e:
            if isinstance(e, MemoryError):
                raise
            raise MemoryError(f"Failed to save context: {str(e)}")
    
    def load_context(self) -> Dict[str, Any]:
        """
        Load context from persistent storage.
        
        Returns:
            Context dictionary loaded from storage
            
        Raises:
            MemoryError: If context cannot be loaded
        """
        try:
            with self._lock:
                if not self.context_file.exists():
                    return {}
                
                with open(self.context_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, dict):
                    raise MemoryError("Invalid context file format: expected dictionary")
                
                return data
                
        except Exception as e:
            if isinstance(e, MemoryError):
                raise
            raise MemoryError(f"Failed to load context: {str(e)}")
    
    def clear_storage(self) -> None:
        """
        Clear all persistent storage files.
        
        Raises:
            MemoryError: If storage cannot be cleared
        """
        try:
            with self._lock:
                # Remove messages file
                if self.messages_file.exists():
                    self.messages_file.unlink()
                
                # Remove context file
                if self.context_file.exists():
                    self.context_file.unlink()
                
        except Exception as e:
            raise MemoryError(f"Failed to clear storage: {str(e)}")
    
    def backup_storage(self, backup_dir: str) -> None:
        """
        Create a backup of the current storage.
        
        Args:
            backup_dir: Directory to store the backup
            
        Raises:
            MemoryError: If backup cannot be created
        """
        if not isinstance(backup_dir, str):
            raise MemoryError("Backup directory must be a string")
        
        try:
            with self._lock:
                backup_path = Path(backup_dir)
                backup_path.mkdir(parents=True, exist_ok=True)
                
                # Backup messages file
                if self.messages_file.exists():
                    backup_messages = backup_path / "messages.json"
                    shutil.copy2(str(self.messages_file), str(backup_messages))
                
                # Backup context file
                if self.context_file.exists():
                    backup_context = backup_path / "context.json"
                    shutil.copy2(str(self.context_file), str(backup_context))
                
        except Exception as e:
            raise MemoryError(f"Failed to create backup: {str(e)}")
    
    def restore_from_backup(self, backup_dir: str) -> None:
        """
        Restore storage from a backup.
        
        Args:
            backup_dir: Directory containing the backup
            
        Raises:
            MemoryError: If restore cannot be completed
        """
        if not isinstance(backup_dir, str):
            raise MemoryError("Backup directory must be a string")
        
        try:
            with self._lock:
                backup_path = Path(backup_dir)
                if not backup_path.exists():
                    raise MemoryError(f"Backup directory does not exist: {backup_dir}")
                
                # Restore messages file
                backup_messages = backup_path / "messages.json"
                if backup_messages.exists():
                    shutil.copy2(str(backup_messages), str(self.messages_file))
                
                # Restore context file
                backup_context = backup_path / "context.json"
                if backup_context.exists():
                    shutil.copy2(str(backup_context), str(self.context_file))
                
        except Exception as e:
            if isinstance(e, MemoryError):
                raise
            raise MemoryError(f"Failed to restore from backup: {str(e)}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about the storage files.
        
        Returns:
            Dictionary containing storage information
            
        Raises:
            MemoryError: If storage info cannot be retrieved
        """
        try:
            with self._lock:
                info = {
                    "storage_dir": str(self.storage_dir),
                    "messages_file": {
                        "path": str(self.messages_file),
                        "exists": self.messages_file.exists(),
                        "size": 0,
                        "modified": None
                    },
                    "context_file": {
                        "path": str(self.context_file),
                        "exists": self.context_file.exists(),
                        "size": 0,
                        "modified": None
                    }
                }
                
                # Get messages file info
                if self.messages_file.exists():
                    stat = self.messages_file.stat()
                    info["messages_file"]["size"] = stat.st_size
                    info["messages_file"]["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                
                # Get context file info
                if self.context_file.exists():
                    stat = self.context_file.stat()
                    info["context_file"]["size"] = stat.st_size
                    info["context_file"]["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                
                return info
                
        except Exception as e:
            raise MemoryError(f"Failed to get storage info: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        # Perform any necessary cleanup
        pass