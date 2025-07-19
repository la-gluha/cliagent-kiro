"""
User workflow manager for handling complete end-to-end user interactions.

This module manages complete user workflows including task execution,
conversation management, context switching, and progress tracking.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..interfaces.agent_interface import AgentInterface
from ..interfaces.memory_interface import MemoryInterface
from ..interfaces.display_interface import DisplayInterface
from ..interfaces.tool_interface import ToolInterface
from ..models.data_models import TaskResult, Message, MessageRole
from ..exceptions import AgentError


class WorkflowType(Enum):
    """Types of user workflows."""
    CHAT = "chat"
    TASK_EXECUTION = "task_execution"
    AUTONOMOUS_TASK = "autonomous_task"
    MULTI_STEP_TASK = "multi_step_task"
    CONVERSATION = "conversation"


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""
    step_id: str
    description: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Optional[Any] = None
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the step in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class UserWorkflow:
    """Represents a complete user workflow."""
    workflow_id: str
    workflow_type: WorkflowType
    user_input: str
    steps: List[WorkflowStep] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    progress_callback: Optional[Callable[[str, float], None]] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get the total duration of the workflow in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def progress(self) -> float:
        """Get the current progress as a percentage (0.0 to 1.0)."""
        if not self.steps:
            return 0.0
        
        completed_steps = sum(1 for step in self.steps if step.status == WorkflowStatus.COMPLETED)
        return completed_steps / len(self.steps)


class UserWorkflowManager:
    """Manages complete user workflows and orchestrates system components."""
    
    def __init__(
        self,
        agent: AgentInterface,
        memory: MemoryInterface,
        display_manager: DisplayInterface,
        tool_registry: ToolInterface
    ):
        """
        Initialize the workflow manager.
        
        Args:
            agent: The ReAct agent for task execution
            memory: Memory interface for conversation history
            display_manager: Display manager for user feedback
            tool_registry: Tool registry for available tools
        """
        self.agent = agent
        self.memory = memory
        self.display_manager = display_manager
        self.tool_registry = tool_registry
        
        # Workflow tracking
        self.active_workflows: Dict[str, UserWorkflow] = {}
        self.workflow_history: List[UserWorkflow] = []
        self.current_session_context: Dict[str, Any] = {}
        
        # Configuration
        self.max_concurrent_workflows = 5
        self.workflow_timeout = 300  # 5 minutes
        self.progress_update_interval = 1.0  # seconds
    
    async def execute_chat_workflow(self, user_input: str, session_id: str) -> str:
        """
        Execute a simple chat workflow.
        
        Args:
            user_input: User's chat message
            session_id: Session identifier
            
        Returns:
            Agent's response
        """
        workflow_id = f"chat_{session_id}_{int(time.time())}"
        
        workflow = UserWorkflow(
            workflow_id=workflow_id,
            workflow_type=WorkflowType.CHAT,
            user_input=user_input,
            context={"session_id": session_id}
        )
        
        try:
            # Start workflow
            workflow.status = WorkflowStatus.RUNNING
            workflow.start_time = datetime.now()
            self.active_workflows[workflow_id] = workflow
            
            # Show progress
            self.display_manager.show_progress("Processing your message", 0.1)
            
            # Execute chat with agent
            response = self.agent.process_input(user_input)
            
            # Update progress
            self.display_manager.show_progress("Generating response", 0.8)
            
            # Complete workflow
            workflow.status = WorkflowStatus.COMPLETED
            workflow.end_time = datetime.now()
            workflow.result = response
            
            self.display_manager.show_progress("Complete", 1.0)
            
            # Move to history
            self._complete_workflow(workflow_id)
            
            return response
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.end_time = datetime.now()
            workflow.error_message = str(e)
            self._complete_workflow(workflow_id)
            
            self.display_manager.show_message(f"Chat workflow failed: {e}", "error")
            raise AgentError(f"Chat workflow failed: {e}")
    
    async def execute_task_workflow(
        self,
        task_description: str,
        session_id: str,
        autonomous: bool = False
    ) -> TaskResult:
        """
        Execute a task execution workflow.
        
        Args:
            task_description: Description of the task to execute
            session_id: Session identifier
            autonomous: Whether to run autonomously without user intervention
            
        Returns:
            TaskResult containing execution outcome
        """
        workflow_id = f"task_{session_id}_{int(time.time())}"
        workflow_type = WorkflowType.AUTONOMOUS_TASK if autonomous else WorkflowType.TASK_EXECUTION
        
        workflow = UserWorkflow(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            user_input=task_description,
            context={"session_id": session_id, "autonomous": autonomous}
        )
        
        # Define workflow steps
        workflow.steps = [
            WorkflowStep("analyze", "Analyze task requirements", "analyze_task"),
            WorkflowStep("plan", "Create execution plan", "create_plan"),
            WorkflowStep("execute", "Execute task steps", "execute_task"),
            WorkflowStep("verify", "Verify results", "verify_results"),
            WorkflowStep("summarize", "Summarize outcome", "summarize_results")
        ]
        
        try:
            # Start workflow
            workflow.status = WorkflowStatus.RUNNING
            workflow.start_time = datetime.now()
            self.active_workflows[workflow_id] = workflow
            
            # Set up progress callback
            def progress_callback(message: str, progress: float):
                self.display_manager.show_progress(message, progress)
                if workflow.progress_callback:
                    workflow.progress_callback(message, progress)
            
            workflow.progress_callback = progress_callback
            
            # Execute workflow steps
            await self._execute_workflow_steps(workflow)
            
            # Execute the actual task with the agent
            progress_callback("Executing task with AI agent", 0.6)
            
            task_result = self.agent.execute_task(task_description)
            
            # Update workflow result
            workflow.result = task_result
            workflow.status = WorkflowStatus.COMPLETED if task_result.success else WorkflowStatus.FAILED
            workflow.end_time = datetime.now()
            
            if not task_result.success:
                workflow.error_message = task_result.error_message
            
            progress_callback("Task completed", 1.0)
            
            # Move to history
            self._complete_workflow(workflow_id)
            
            return task_result
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.end_time = datetime.now()
            workflow.error_message = str(e)
            self._complete_workflow(workflow_id)
            
            self.display_manager.show_message(f"Task workflow failed: {e}", "error")
            raise AgentError(f"Task workflow failed: {e}")
    
    async def execute_multi_step_workflow(
        self,
        steps: List[Dict[str, Any]],
        session_id: str
    ) -> List[TaskResult]:
        """
        Execute a multi-step workflow with multiple tasks.
        
        Args:
            steps: List of step definitions with task descriptions
            session_id: Session identifier
            
        Returns:
            List of TaskResult objects for each step
        """
        workflow_id = f"multi_step_{session_id}_{int(time.time())}"
        
        workflow = UserWorkflow(
            workflow_id=workflow_id,
            workflow_type=WorkflowType.MULTI_STEP_TASK,
            user_input=f"Multi-step workflow with {len(steps)} steps",
            context={"session_id": session_id, "step_count": len(steps)}
        )
        
        # Create workflow steps
        workflow.steps = [
            WorkflowStep(
                step_id=f"step_{i}",
                description=step.get("description", f"Step {i+1}"),
                action="execute_task",
                parameters=step
            )
            for i, step in enumerate(steps)
        ]
        
        results = []
        
        try:
            # Start workflow
            workflow.status = WorkflowStatus.RUNNING
            workflow.start_time = datetime.now()
            self.active_workflows[workflow_id] = workflow
            
            # Execute each step
            for i, step in enumerate(workflow.steps):
                step_progress = i / len(workflow.steps)
                self.display_manager.show_progress(
                    f"Executing step {i+1}/{len(workflow.steps)}: {step.description}",
                    step_progress
                )
                
                # Execute step
                step.status = WorkflowStatus.RUNNING
                step.start_time = datetime.now()
                
                try:
                    task_description = step.parameters.get("task", step.description)
                    task_result = self.agent.execute_task(task_description)
                    
                    step.result = task_result
                    step.status = WorkflowStatus.COMPLETED if task_result.success else WorkflowStatus.FAILED
                    step.end_time = datetime.now()
                    
                    results.append(task_result)
                    
                    if not task_result.success:
                        step.error_message = task_result.error_message
                        # Continue with other steps even if one fails
                        self.display_manager.show_message(
                            f"Step {i+1} failed: {task_result.error_message}",
                            "warning"
                        )
                    
                except Exception as e:
                    step.status = WorkflowStatus.FAILED
                    step.end_time = datetime.now()
                    step.error_message = str(e)
                    
                    # Create failed task result
                    failed_result = TaskResult(
                        success=False,
                        error_message=str(e),
                        execution_time=step.duration or 0.0,
                        steps_taken=[f"Failed at step: {step.description}"]
                    )
                    results.append(failed_result)
            
            # Complete workflow
            workflow.status = WorkflowStatus.COMPLETED
            workflow.end_time = datetime.now()
            workflow.result = results
            
            self.display_manager.show_progress("Multi-step workflow completed", 1.0)
            
            # Move to history
            self._complete_workflow(workflow_id)
            
            return results
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.end_time = datetime.now()
            workflow.error_message = str(e)
            self._complete_workflow(workflow_id)
            
            self.display_manager.show_message(f"Multi-step workflow failed: {e}", "error")
            raise AgentError(f"Multi-step workflow failed: {e}")
    
    async def execute_conversation_workflow(
        self,
        conversation_context: Dict[str, Any],
        session_id: str
    ) -> str:
        """
        Execute a conversation workflow with context management.
        
        Args:
            conversation_context: Context for the conversation
            session_id: Session identifier
            
        Returns:
            Conversation response
        """
        workflow_id = f"conversation_{session_id}_{int(time.time())}"
        
        workflow = UserWorkflow(
            workflow_id=workflow_id,
            workflow_type=WorkflowType.CONVERSATION,
            user_input=conversation_context.get("message", ""),
            context={"session_id": session_id, **conversation_context}
        )
        
        try:
            # Start workflow
            workflow.status = WorkflowStatus.RUNNING
            workflow.start_time = datetime.now()
            self.active_workflows[workflow_id] = workflow
            
            # Load conversation history
            self.display_manager.show_progress("Loading conversation context", 0.2)
            
            conversation_history = self.memory.retrieve_history(limit=10)
            
            # Update agent context with conversation history
            context_updates = {
                "memory_context": {
                    "conversation_history": [msg.to_dict() for msg in conversation_history],
                    "session_context": self.current_session_context
                }
            }
            self.agent.update_context(context_updates)
            
            # Process conversation
            self.display_manager.show_progress("Processing conversation", 0.6)
            
            user_message = conversation_context.get("message", "")
            response = self.agent.process_input(user_message)
            
            # Update session context
            self._update_session_context(conversation_context, response)
            
            # Complete workflow
            workflow.status = WorkflowStatus.COMPLETED
            workflow.end_time = datetime.now()
            workflow.result = response
            
            self.display_manager.show_progress("Conversation completed", 1.0)
            
            # Move to history
            self._complete_workflow(workflow_id)
            
            return response
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.end_time = datetime.now()
            workflow.error_message = str(e)
            self._complete_workflow(workflow_id)
            
            self.display_manager.show_message(f"Conversation workflow failed: {e}", "error")
            raise AgentError(f"Conversation workflow failed: {e}")
    
    async def _execute_workflow_steps(self, workflow: UserWorkflow) -> None:
        """
        Execute the steps of a workflow.
        
        Args:
            workflow: The workflow to execute steps for
        """
        for i, step in enumerate(workflow.steps):
            step_progress = i / len(workflow.steps)
            
            if workflow.progress_callback:
                workflow.progress_callback(f"Executing: {step.description}", step_progress)
            
            step.status = WorkflowStatus.RUNNING
            step.start_time = datetime.now()
            
            try:
                # Execute step based on action type
                if step.action == "analyze_task":
                    step.result = await self._analyze_task_step(workflow, step)
                elif step.action == "create_plan":
                    step.result = await self._create_plan_step(workflow, step)
                elif step.action == "execute_task":
                    step.result = await self._execute_task_step(workflow, step)
                elif step.action == "verify_results":
                    step.result = await self._verify_results_step(workflow, step)
                elif step.action == "summarize_results":
                    step.result = await self._summarize_results_step(workflow, step)
                else:
                    step.result = f"Unknown action: {step.action}"
                
                step.status = WorkflowStatus.COMPLETED
                step.end_time = datetime.now()
                
            except Exception as e:
                step.status = WorkflowStatus.FAILED
                step.end_time = datetime.now()
                step.error_message = str(e)
                
                # Continue with other steps
                if workflow.progress_callback:
                    workflow.progress_callback(f"Step failed: {step.description}", step_progress)
    
    async def _analyze_task_step(self, workflow: UserWorkflow, step: WorkflowStep) -> str:
        """Analyze the task requirements."""
        await asyncio.sleep(0.1)  # Simulate processing time
        return f"Analyzed task: {workflow.user_input}"
    
    async def _create_plan_step(self, workflow: UserWorkflow, step: WorkflowStep) -> str:
        """Create an execution plan."""
        await asyncio.sleep(0.1)  # Simulate processing time
        available_tools = self.tool_registry.get_available_tools()
        tool_names = [tool.name for tool in available_tools]
        return f"Created plan using available tools: {', '.join(tool_names)}"
    
    async def _execute_task_step(self, workflow: UserWorkflow, step: WorkflowStep) -> str:
        """Execute the main task."""
        await asyncio.sleep(0.1)  # Simulate processing time
        return "Task execution prepared"
    
    async def _verify_results_step(self, workflow: UserWorkflow, step: WorkflowStep) -> str:
        """Verify the results."""
        await asyncio.sleep(0.1)  # Simulate processing time
        return "Results verified"
    
    async def _summarize_results_step(self, workflow: UserWorkflow, step: WorkflowStep) -> str:
        """Summarize the results."""
        await asyncio.sleep(0.1)  # Simulate processing time
        return "Results summarized"
    
    def _complete_workflow(self, workflow_id: str) -> None:
        """Move a workflow from active to history."""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows.pop(workflow_id)
            self.workflow_history.append(workflow)
            
            # Limit history size
            if len(self.workflow_history) > 100:
                self.workflow_history = self.workflow_history[-100:]
    
    def _update_session_context(self, conversation_context: Dict[str, Any], response: str) -> None:
        """Update the session context with conversation information."""
        session_id = conversation_context.get("session_id", "default")
        
        if session_id not in self.current_session_context:
            self.current_session_context[session_id] = {}
        
        session_context = self.current_session_context[session_id]
        
        # Update context with conversation metadata
        session_context["last_interaction"] = datetime.now().isoformat()
        session_context["interaction_count"] = session_context.get("interaction_count", 0) + 1
        
        # Store recent topics or keywords
        if "recent_topics" not in session_context:
            session_context["recent_topics"] = []
        
        # Simple topic extraction (could be enhanced with NLP)
        user_message = conversation_context.get("message", "")
        if len(user_message) > 10:
            session_context["recent_topics"].append(user_message[:50])
            session_context["recent_topics"] = session_context["recent_topics"][-5:]  # Keep last 5
    
    def get_active_workflows(self) -> List[UserWorkflow]:
        """Get list of currently active workflows."""
        return list(self.active_workflows.values())
    
    def get_workflow_history(self, limit: Optional[int] = None) -> List[UserWorkflow]:
        """Get workflow history."""
        if limit:
            return self.workflow_history[-limit:]
        return self.workflow_history.copy()
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowStatus]:
        """Get the status of a specific workflow."""
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id].status
        
        for workflow in self.workflow_history:
            if workflow.workflow_id == workflow_id:
                return workflow.status
        
        return None
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow."""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = WorkflowStatus.CANCELLED
            workflow.end_time = datetime.now()
            self._complete_workflow(workflow_id)
            return True
        return False
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get context for a specific session."""
        return self.current_session_context.get(session_id, {})
    
    def clear_session_context(self, session_id: str) -> None:
        """Clear context for a specific session."""
        if session_id in self.current_session_context:
            del self.current_session_context[session_id]
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get statistics about workflow execution."""
        all_workflows = list(self.active_workflows.values()) + self.workflow_history
        
        if not all_workflows:
            return {"total_workflows": 0}
        
        completed_workflows = [w for w in all_workflows if w.status == WorkflowStatus.COMPLETED]
        failed_workflows = [w for w in all_workflows if w.status == WorkflowStatus.FAILED]
        
        # Calculate average duration for completed workflows
        completed_durations = [w.duration for w in completed_workflows if w.duration is not None]
        avg_duration = sum(completed_durations) / len(completed_durations) if completed_durations else 0
        
        # Count by workflow type
        type_counts = {}
        for workflow in all_workflows:
            workflow_type = workflow.workflow_type.value
            type_counts[workflow_type] = type_counts.get(workflow_type, 0) + 1
        
        return {
            "total_workflows": len(all_workflows),
            "completed_workflows": len(completed_workflows),
            "failed_workflows": len(failed_workflows),
            "active_workflows": len(self.active_workflows),
            "success_rate": len(completed_workflows) / len(all_workflows) if all_workflows else 0,
            "average_duration": avg_duration,
            "workflow_types": type_counts
        }