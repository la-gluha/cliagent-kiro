"""
Core data models for the AI Agent System.

This module contains the fundamental data structures used throughout the system,
including Message, TaskResult, ToolInfo, and AgentState classes with validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import uuid
import json


class MessageRole(Enum):
    """Enumeration for message roles."""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class ToolCategory(Enum):
    """Enumeration for tool categories."""
    FILE_OPERATIONS = "file_operations"
    WEB_SEARCH = "web_search"
    CALCULATION = "calculation"
    SYSTEM = "system"
    CUSTOM = "custom"


class ModelType(Enum):
    """Enumeration for model types."""
    OPENAI = "openai"
    LOCAL = "local"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


@dataclass
class Message:
    """
    Represents a message in the conversation history.
    
    Attributes:
        id: Unique identifier for the message
        content: The message content
        role: The role of the message sender (user, agent, system)
        timestamp: When the message was created
        metadata: Additional metadata for the message
    """
    content: str
    role: MessageRole
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the message after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate the message data.
        
        Raises:
            ValueError: If validation fails
        """
        if not self.content or not isinstance(self.content, str):
            raise ValueError("Message content must be a non-empty string")
        
        if not isinstance(self.role, MessageRole):
            if isinstance(self.role, str):
                try:
                    self.role = MessageRole(self.role)
                except ValueError:
                    raise ValueError(f"Invalid message role: {self.role}")
            else:
                raise ValueError("Message role must be a MessageRole enum or valid string")
        
        if not isinstance(self.timestamp, datetime):
            raise ValueError("Message timestamp must be a datetime object")
        
        if not isinstance(self.metadata, dict):
            raise ValueError("Message metadata must be a dictionary")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message from a dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            role=MessageRole(data["role"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class TaskResult:
    """
    Represents the result of a task execution.
    
    Attributes:
        success: Whether the task completed successfully
        result: The result data from the task
        error_message: Error message if the task failed
        execution_time: Time taken to execute the task in seconds
        steps_taken: List of steps taken during execution
    """
    success: bool
    result: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    steps_taken: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate the task result after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate the task result data.
        
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(self.success, bool):
            raise ValueError("TaskResult success must be a boolean")
        
        if self.error_message is not None and not isinstance(self.error_message, str):
            raise ValueError("TaskResult error_message must be a string or None")
        
        if not isinstance(self.execution_time, (int, float)) or self.execution_time < 0:
            raise ValueError("TaskResult execution_time must be a non-negative number")
        
        if not isinstance(self.steps_taken, list):
            raise ValueError("TaskResult steps_taken must be a list")
        
        # Validate that all steps are strings
        for step in self.steps_taken:
            if not isinstance(step, str):
                raise ValueError("All steps in steps_taken must be strings")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the task result to a dictionary for serialization."""
        return {
            "success": self.success,
            "result": self.result,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "steps_taken": self.steps_taken
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResult":
        """Create a TaskResult from a dictionary."""
        return cls(
            success=data["success"],
            result=data.get("result"),
            error_message=data.get("error_message"),
            execution_time=data.get("execution_time", 0.0),
            steps_taken=data.get("steps_taken", [])
        )


@dataclass
class ToolInfo:
    """
    Information about an available tool.
    
    Attributes:
        name: The name of the tool
        description: Description of what the tool does
        parameters: Dictionary describing the tool's parameters
        category: The category this tool belongs to
        enabled: Whether the tool is currently enabled
    """
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    category: ToolCategory = ToolCategory.CUSTOM
    enabled: bool = True
    
    def __post_init__(self):
        """Validate the tool info after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate the tool info data.
        
        Raises:
            ValueError: If validation fails
        """
        if not self.name or not isinstance(self.name, str):
            raise ValueError("ToolInfo name must be a non-empty string")
        
        if not self.description or not isinstance(self.description, str):
            raise ValueError("ToolInfo description must be a non-empty string")
        
        if not isinstance(self.parameters, dict):
            raise ValueError("ToolInfo parameters must be a dictionary")
        
        if not isinstance(self.category, ToolCategory):
            if isinstance(self.category, str):
                try:
                    self.category = ToolCategory(self.category)
                except ValueError:
                    raise ValueError(f"Invalid tool category: {self.category}")
            else:
                raise ValueError("ToolInfo category must be a ToolCategory enum or valid string")
        
        if not isinstance(self.enabled, bool):
            raise ValueError("ToolInfo enabled must be a boolean")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool info to a dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "category": self.category.value,
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolInfo":
        """Create a ToolInfo from a dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters", {}),
            category=ToolCategory(data.get("category", "custom")),
            enabled=data.get("enabled", True)
        )


@dataclass
class ModelInfo:
    """
    Information about an available language model.
    
    Attributes:
        name: The name of the model
        model_type: The type/provider of the model
        description: Description of the model's capabilities
        max_tokens: Maximum tokens the model can handle
        supports_streaming: Whether the model supports streaming responses
        api_endpoint: API endpoint for the model (if applicable)
        version: Version of the model
        enabled: Whether the model is currently enabled
    """
    name: str
    model_type: ModelType
    description: str = ""
    max_tokens: int = 4096
    supports_streaming: bool = False
    api_endpoint: Optional[str] = None
    version: str = "1.0"
    enabled: bool = True
    
    def __post_init__(self):
        """Validate the model info after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate the model info data.
        
        Raises:
            ValueError: If validation fails
        """
        if not self.name or not isinstance(self.name, str):
            raise ValueError("ModelInfo name must be a non-empty string")
        
        if not isinstance(self.model_type, ModelType):
            if isinstance(self.model_type, str):
                try:
                    self.model_type = ModelType(self.model_type)
                except ValueError:
                    raise ValueError(f"Invalid model type: {self.model_type}")
            else:
                raise ValueError("ModelInfo model_type must be a ModelType enum or valid string")
        
        if not isinstance(self.description, str):
            raise ValueError("ModelInfo description must be a string")
        
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("ModelInfo max_tokens must be a positive integer")
        
        if not isinstance(self.supports_streaming, bool):
            raise ValueError("ModelInfo supports_streaming must be a boolean")
        
        if self.api_endpoint is not None and not isinstance(self.api_endpoint, str):
            raise ValueError("ModelInfo api_endpoint must be a string or None")
        
        if not isinstance(self.version, str):
            raise ValueError("ModelInfo version must be a string")
        
        if not isinstance(self.enabled, bool):
            raise ValueError("ModelInfo enabled must be a boolean")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model info to a dictionary for serialization."""
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "description": self.description,
            "max_tokens": self.max_tokens,
            "supports_streaming": self.supports_streaming,
            "api_endpoint": self.api_endpoint,
            "version": self.version,
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """Create a ModelInfo from a dictionary."""
        return cls(
            name=data["name"],
            model_type=ModelType(data["model_type"]),
            description=data.get("description", ""),
            max_tokens=data.get("max_tokens", 4096),
            supports_streaming=data.get("supports_streaming", False),
            api_endpoint=data.get("api_endpoint"),
            version=data.get("version", "1.0"),
            enabled=data.get("enabled", True)
        )


@dataclass
class AgentState:
    """
    Represents the current state of an agent.
    
    Attributes:
        current_task: The task currently being executed
        reasoning_history: History of reasoning steps
        action_history: History of actions taken
        context: Current context information
        is_active: Whether the agent is currently active
    """
    current_task: Optional[str] = None
    reasoning_history: List[str] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = False
    
    def __post_init__(self):
        """Validate the agent state after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate the agent state data.
        
        Raises:
            ValueError: If validation fails
        """
        if self.current_task is not None and not isinstance(self.current_task, str):
            raise ValueError("AgentState current_task must be a string or None")
        
        if not isinstance(self.reasoning_history, list):
            raise ValueError("AgentState reasoning_history must be a list")
        
        # Validate that all reasoning steps are strings
        for step in self.reasoning_history:
            if not isinstance(step, str):
                raise ValueError("All items in reasoning_history must be strings")
        
        if not isinstance(self.action_history, list):
            raise ValueError("AgentState action_history must be a list")
        
        # Validate that all actions are dictionaries
        for action in self.action_history:
            if not isinstance(action, dict):
                raise ValueError("All items in action_history must be dictionaries")
        
        if not isinstance(self.context, dict):
            raise ValueError("AgentState context must be a dictionary")
        
        if not isinstance(self.is_active, bool):
            raise ValueError("AgentState is_active must be a boolean")
    
    def add_reasoning_step(self, step: str) -> None:
        """Add a reasoning step to the history."""
        if not isinstance(step, str):
            raise ValueError("Reasoning step must be a string")
        self.reasoning_history.append(step)
    
    def add_action(self, action: Dict[str, Any]) -> None:
        """Add an action to the history."""
        if not isinstance(action, dict):
            raise ValueError("Action must be a dictionary")
        self.action_history.append(action)
    
    def clear_history(self) -> None:
        """Clear the reasoning and action history."""
        self.reasoning_history.clear()
        self.action_history.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the agent state to a dictionary for serialization."""
        return {
            "current_task": self.current_task,
            "reasoning_history": self.reasoning_history,
            "action_history": self.action_history,
            "context": self.context,
            "is_active": self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Create an AgentState from a dictionary."""
        return cls(
            current_task=data.get("current_task"),
            reasoning_history=data.get("reasoning_history", []),
            action_history=data.get("action_history", []),
            context=data.get("context", {}),
            is_active=data.get("is_active", False)
        )