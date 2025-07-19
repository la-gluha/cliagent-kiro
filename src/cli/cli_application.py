"""Main CLI application implementing the CLIInterface."""

import sys
import os
from typing import Optional, Dict, Any
from ..interfaces.cli_interface import CLIInterface
from ..interfaces.display_interface import DisplayInterface
from .command_parser import CommandParser, ParsedCommand
from .session_manager import SessionManager


class CLIApplication(CLIInterface):
    """Main CLI application for the AI Agent System."""
    
    def __init__(self, display_manager: Optional[DisplayInterface] = None):
        """
        Initialize the CLI application.
        
        Args:
            display_manager: Display manager for output formatting
        """
        self.command_parser = CommandParser()
        self.session_manager = SessionManager()
        self.display_manager = display_manager
        self.is_running = False
        self.welcome_message = """
╔══════════════════════════════════════════════════════════════╗
║                    AI Agent System CLI                       ║
║                                                              ║
║  Welcome to the ReAct-based AI Agent System!                ║
║  Type 'help' for available commands or start chatting.      ║
║                                                              ║
║  Press Ctrl+C or type 'exit' to quit.                       ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    def start_session(self) -> None:
        """Start a new CLI session."""
        # Create a new session
        session = self.session_manager.create_session()
        self.is_running = True
        
        # Display welcome message
        self.display_output(self.welcome_message, "info")
        
        # Show session info
        session_info = self.session_manager.get_session_info()
        self.display_output(f"Session {session_info['session_id']} started.", "success")
        
        # Start the main input loop
        self._run_input_loop()
    
    def handle_input(self, user_input: str) -> None:
        """
        Handle user input and process commands.
        
        Args:
            user_input: Raw user input string
        """
        if not user_input.strip():
            return
        
        try:
            # Parse the command
            parsed_command = self.command_parser.parse(user_input)
            
            # Validate the command
            is_valid, error_message = self.command_parser.validate_command(parsed_command)
            
            if not is_valid:
                self.display_output(f"Error: {error_message}", "error")
                
                # Try to suggest a similar command
                suggestion = self.command_parser.suggest_command(parsed_command.command)
                if suggestion:
                    self.display_output(f"Did you mean '{suggestion}'?", "suggestion")
                    self.display_output(f"Type 'help {suggestion}' for more information.", "info")
                else:
                    self.display_output("Type 'help' to see all available commands.", "info")
                return
            
            # Execute the command
            self._execute_command(parsed_command)
            
            # Update session activity after successful execution
            self.session_manager.update_session_activity(user_input)
            
        except ValueError as e:
            self.display_output(f"Parse error: {e}", "error")
        except Exception as e:
            self.display_output(f"Unexpected error: {e}", "error")
    
    def display_output(self, content: str, output_type: str = "info") -> None:
        """
        Display output to the user.
        
        Args:
            content: Content to display
            output_type: Type of output (info, error, success, warning, suggestion)
        """
        if self.display_manager:
            self.display_manager.show_message(content, output_type)
        else:
            # Fallback to simple print with basic formatting
            prefix_map = {
                "error": "❌ ERROR: ",
                "success": "✅ ",
                "warning": "⚠️  WARNING: ",
                "suggestion": "💡 ",
                "info": ""
            }
            prefix = prefix_map.get(output_type, "")
            print(f"{prefix}{content}")
    
    def show_progress(self, message: str, progress: float) -> None:
        """
        Show progress information to the user.
        
        Args:
            message: Progress message
            progress: Progress value between 0.0 and 1.0
        """
        if self.display_manager:
            self.display_manager.show_progress(message, progress)
        else:
            # Simple progress display
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            percentage = int(progress * 100)
            print(f"\r{message}: [{bar}] {percentage}%", end="", flush=True)
            if progress >= 1.0:
                print()  # New line when complete
    
    def shutdown(self) -> None:
        """Gracefully shutdown the CLI session."""
        self.is_running = False
        self.session_manager.end_session()
        self.display_output("Goodbye! 👋", "info")
    
    def _run_input_loop(self) -> None:
        """Run the main input loop with enhanced interactive features."""
        interrupt_count = 0
        
        while self.is_running:
            try:
                # Get user input with a prompt
                session = self.session_manager.get_current_session()
                session_id = session.session_id if session else "unknown"
                prompt = f"agent[{session_id}]> "
                
                user_input = input(prompt)
                
                # Reset interrupt count on successful input
                interrupt_count = 0
                
                # Handle empty input gracefully
                if not user_input.strip():
                    continue
                
                self.handle_input(user_input)
                
            except KeyboardInterrupt:
                print()  # New line after Ctrl+C
                interrupt_count += 1
                
                if interrupt_count == 1:
                    self.display_output("Interrupted. Press Ctrl+C again to exit or type 'exit' to quit.", "warning")
                elif interrupt_count >= 2:
                    self.display_output("Exiting due to repeated interrupts...", "info")
                    self.shutdown()
                    break
                    
            except EOFError:
                print()  # New line after Ctrl+D
                self.display_output("EOF received. Shutting down gracefully...", "info")
                self.shutdown()
                break
                
            except Exception as e:
                self.display_output(f"Unexpected input error: {e}", "error")
                self.display_output("Please try again or type 'help' for assistance.", "info")
    
    def _execute_command(self, parsed_command: ParsedCommand) -> None:
        """
        Execute a parsed and validated command.
        
        Args:
            parsed_command: The parsed command to execute
        """
        command = parsed_command.command
        args = parsed_command.args
        kwargs = parsed_command.kwargs
        
        if command in ['exit', 'quit']:
            self.shutdown()
        
        elif command == 'help':
            if args:
                help_text = self.command_parser.get_help_text(args[0])
            else:
                help_text = self.command_parser.get_help_text()
            self.display_output(help_text, "info")
        
        elif command == 'clear':
            # Clear the screen
            os.system('cls' if os.name == 'nt' else 'clear')
        
        elif command == 'history':
            history = self.session_manager.get_command_history()
            if not history:
                self.display_output("No command history.", "info")
            else:
                self.display_output("Command History:", "info")
                for i, cmd in enumerate(history[-10:], 1):  # Show last 10 commands
                    self.display_output(f"  {i:2d}. {cmd}", "info")
        
        elif command == 'task':
            task_description = ' '.join(args)
            self._execute_task_workflow(task_description)
        
        elif command == 'chat':
            if args:
                message = ' '.join(args)
                self._execute_chat_workflow(message)
            else:
                self.display_output("Chat mode activated. Type your message:", "info")
                self.display_output("Usage: chat <your message>", "info")
        
        elif command == 'tools':
            self._show_available_tools()
        
        elif command == 'workflows':
            self._show_workflow_status()
        
        elif command == 'multi-task':
            if args:
                self._execute_multi_step_workflow(args)
            else:
                self.display_output("Multi-step task execution:", "info")
                self.display_output("Usage: multi-task <task1> | <task2> | <task3>", "info")
                self.display_output("Example: multi-task search for Python tutorials | count the results | summarize findings", "info")
        
        elif command == 'conversation':
            if args:
                message = ' '.join(args)
                self._execute_conversation_workflow(message)
            else:
                self.display_output("Conversation mode:", "info")
                self.display_output("Usage: conversation <message>", "info")
        
        elif command == 'memory':
            session_info = self.session_manager.get_session_info()
            self.display_output("Session Information:", "info")
            for key, value in session_info.items():
                self.display_output(f"  {key}: {value}", "info")
        
        elif command == 'config':
            if not args:
                self.display_output("Configuration management:", "info")
                self.display_output("  config <key>        - Show configuration value", "info")
                self.display_output("  config <key> <value> - Set configuration value", "info")
                self.display_output("Note: Configuration management not yet fully implemented.", "warning")
            elif len(args) == 1:
                key = args[0]
                value = self.session_manager.get_session_context(f"config_{key}")
                if value is not None:
                    self.display_output(f"{key}: {value}", "info")
                else:
                    self.display_output(f"Configuration key '{key}' not found.", "warning")
            else:
                key, value = args[0], args[1]
                self.session_manager.set_session_context(f"config_{key}", value)
                self.display_output(f"Set {key} = {value}", "success")
        
        else:
            self.display_output(f"Command '{command}' recognized but not yet implemented.", "warning")
    
    def _execute_task_workflow(self, task_description: str) -> None:
        """Execute a task workflow using the workflow manager."""
        if not hasattr(self, 'workflow_manager') or not self.workflow_manager:
            self.display_output("Task execution not available - workflow manager not initialized.", "error")
            return
        
        if not task_description.strip():
            self.display_output("Please provide a task description.", "error")
            self.display_output("Usage: task <task description>", "info")
            return
        
        try:
            session = self.session_manager.get_current_session()
            session_id = session.session_id if session else "default"
            
            self.display_output(f"🚀 Executing task: {task_description}", "info")
            
            # Execute task workflow asynchronously
            import asyncio
            
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in a loop, we need to run in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.workflow_manager.execute_task_workflow(task_description, session_id, autonomous=True)
                    )
                    result = future.result()
            except RuntimeError:
                # No event loop running, we can use asyncio.run
                result = asyncio.run(
                    self.workflow_manager.execute_task_workflow(task_description, session_id, autonomous=True)
                )
            
            # Display results
            if result.success:
                self.display_output("✅ Task completed successfully!", "success")
                if result.result:
                    self.display_output(f"Result: {result.result}", "info")
                
                if result.steps_taken:
                    self.display_output("Steps taken:", "info")
                    for step in result.steps_taken:
                        self.display_output(f"  • {step}", "info")
                
                self.display_output(f"⏱️  Execution time: {result.execution_time:.2f} seconds", "info")
            else:
                self.display_output("❌ Task failed to complete.", "error")
                if result.error_message:
                    self.display_output(f"Error: {result.error_message}", "error")
        
        except Exception as e:
            self.display_output(f"❌ Task execution failed: {e}", "error")
    
    def _execute_chat_workflow(self, message: str) -> None:
        """Execute a chat workflow using the workflow manager."""
        if not hasattr(self, 'workflow_manager') or not self.workflow_manager:
            self.display_output("Chat not available - workflow manager not initialized.", "error")
            return
        
        if not message.strip():
            self.display_output("Please provide a message.", "error")
            return
        
        try:
            session = self.session_manager.get_current_session()
            session_id = session.session_id if session else "default"
            
            self.display_output(f"💬 Processing: {message}", "info")
            
            # Execute chat workflow asynchronously
            import asyncio
            
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in a loop, we need to run in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.workflow_manager.execute_chat_workflow(message, session_id)
                    )
                    response = future.result()
            except RuntimeError:
                # No event loop running, we can use asyncio.run
                response = asyncio.run(
                    self.workflow_manager.execute_chat_workflow(message, session_id)
                )
            
            # Display response
            self.display_output("🤖 Agent:", "info")
            self.display_output(response, "success")
        
        except Exception as e:
            self.display_output(f"❌ Chat failed: {e}", "error")
    
    def _show_available_tools(self) -> None:
        """Show available tools in the system."""
        if not hasattr(self, 'workflow_manager') or not self.workflow_manager:
            self.display_output("Tools not available - workflow manager not initialized.", "error")
            return
        
        try:
            available_tools = self.workflow_manager.tool_registry.get_available_tools()
            
            if not available_tools:
                self.display_output("No tools available.", "warning")
                return
            
            self.display_output("🔧 Available Tools:", "info")
            for tool in available_tools:
                status = "✅ Enabled" if tool.enabled else "❌ Disabled"
                self.display_output(f"  • {tool.name}: {tool.description} ({status})", "info")
                if hasattr(tool, 'category'):
                    self.display_output(f"    Category: {tool.category.value}", "info")
        
        except Exception as e:
            self.display_output(f"❌ Failed to list tools: {e}", "error")
    
    def _show_workflow_status(self) -> None:
        """Show current workflow status and statistics."""
        if not hasattr(self, 'workflow_manager') or not self.workflow_manager:
            self.display_output("Workflows not available - workflow manager not initialized.", "error")
            return
        
        try:
            # Show active workflows
            active_workflows = self.workflow_manager.get_active_workflows()
            self.display_output("🔄 Active Workflows:", "info")
            
            if not active_workflows:
                self.display_output("  No active workflows", "info")
            else:
                for workflow in active_workflows:
                    progress_percent = int(workflow.progress * 100)
                    self.display_output(
                        f"  • {workflow.workflow_id}: {workflow.workflow_type.value} ({progress_percent}%)",
                        "info"
                    )
            
            # Show workflow statistics
            stats = self.workflow_manager.get_workflow_statistics()
            self.display_output("\n📊 Workflow Statistics:", "info")
            self.display_output(f"  Total workflows: {stats.get('total_workflows', 0)}", "info")
            self.display_output(f"  Completed: {stats.get('completed_workflows', 0)}", "info")
            self.display_output(f"  Failed: {stats.get('failed_workflows', 0)}", "info")
            self.display_output(f"  Success rate: {stats.get('success_rate', 0):.1%}", "info")
            
            if stats.get('average_duration', 0) > 0:
                self.display_output(f"  Average duration: {stats.get('average_duration', 0):.2f}s", "info")
        
        except Exception as e:
            self.display_output(f"❌ Failed to show workflow status: {e}", "error")
    
    def _execute_multi_step_workflow(self, args: list) -> None:
        """Execute a multi-step workflow."""
        if not hasattr(self, 'workflow_manager') or not self.workflow_manager:
            self.display_output("Multi-step tasks not available - workflow manager not initialized.", "error")
            return
        
        try:
            # Parse multi-step tasks separated by |
            task_string = ' '.join(args)
            tasks = [task.strip() for task in task_string.split('|') if task.strip()]
            
            if len(tasks) < 2:
                self.display_output("Please provide at least 2 tasks separated by |", "error")
                self.display_output("Example: multi-task search for Python | count results | summarize", "info")
                return
            
            session = self.session_manager.get_current_session()
            session_id = session.session_id if session else "default"
            
            self.display_output(f"🔄 Executing {len(tasks)} tasks:", "info")
            for i, task in enumerate(tasks, 1):
                self.display_output(f"  {i}. {task}", "info")
            
            # Convert tasks to step format
            steps = [{"task": task, "description": f"Step {i+1}: {task}"} for i, task in enumerate(tasks)]
            
            # Execute multi-step workflow asynchronously
            import asyncio
            
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.workflow_manager.execute_multi_step_workflow(steps, session_id)
                    )
                    results = future.result()
            except RuntimeError:
                results = asyncio.run(
                    self.workflow_manager.execute_multi_step_workflow(steps, session_id)
                )
            
            # Display results
            self.display_output("\n📋 Multi-step Results:", "info")
            for i, result in enumerate(results, 1):
                if result.success:
                    self.display_output(f"  ✅ Step {i}: Completed ({result.execution_time:.2f}s)", "success")
                    if result.result:
                        self.display_output(f"     Result: {str(result.result)[:100]}...", "info")
                else:
                    self.display_output(f"  ❌ Step {i}: Failed", "error")
                    if result.error_message:
                        self.display_output(f"     Error: {result.error_message}", "error")
            
            # Summary
            successful_steps = sum(1 for result in results if result.success)
            total_time = sum(result.execution_time for result in results)
            self.display_output(f"\n📊 Summary: {successful_steps}/{len(results)} steps completed in {total_time:.2f}s", "info")
        
        except Exception as e:
            self.display_output(f"❌ Multi-step workflow failed: {e}", "error")
    
    def _execute_conversation_workflow(self, message: str) -> None:
        """Execute a conversation workflow with enhanced context."""
        if not hasattr(self, 'workflow_manager') or not self.workflow_manager:
            self.display_output("Conversation not available - workflow manager not initialized.", "error")
            return
        
        try:
            session = self.session_manager.get_current_session()
            session_id = session.session_id if session else "default"
            
            # Build conversation context
            conversation_context = {
                "message": message,
                "session_id": session_id,
                "session_info": self.session_manager.get_session_info(),
                "command_history": self.session_manager.get_command_history(limit=5)
            }
            
            self.display_output(f"💭 Starting conversation: {message}", "info")
            
            # Execute conversation workflow asynchronously
            import asyncio
            
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.workflow_manager.execute_conversation_workflow(conversation_context, session_id)
                    )
                    response = future.result()
            except RuntimeError:
                response = asyncio.run(
                    self.workflow_manager.execute_conversation_workflow(conversation_context, session_id)
                )
            
            # Display response with enhanced formatting
            self.display_output("🤖 Agent Response:", "info")
            self.display_output(response, "success")
            
            # Show conversation context if available
            session_context = self.workflow_manager.get_session_context(session_id)
            if session_context and session_context.get("interaction_count", 0) > 1:
                self.display_output(f"💬 Conversation #{session_context['interaction_count']}", "info")
        
        except Exception as e:
            self.display_output(f"❌ Conversation failed: {e}", "error")