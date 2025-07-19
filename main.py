#!/usr/bin/env python3
"""
Main entry point for the AI Agent System.

This module provides the main application entry point that integrates all
components of the ReAct-based AI Agent System including CLI, agents, memory,
tools, and display management.
"""

import sys
import os
import argparse
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cli.cli_application import CLIApplication
from src.agents.react_agent import ReActAgent
from src.memory.conversation_memory import ConversationMemory
from src.memory.memory_persistence import MemoryPersistence
from src.tools.tool_registry import ToolRegistry
from src.tools.basic_tools import FileOperationsTool, CalculatorTool, WebSearchTool
from src.display.display_manager import DisplayManager
from src.models.ai_api_manager import AIAPIManager
from src.models.config_models import AgentConfig, DisplayConfig
from src.exceptions import AgentError, ConfigurationError
from src.workflows.user_workflow_manager import UserWorkflowManager


class AIAgentSystemApp:
    """Main application class that orchestrates all system components."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AI Agent System application.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self.agent_config = None
        self.display_config = None
        self.components = {}
        self.workflow_manager = None
        
    def initialize(self) -> None:
        """Initialize all system components."""
        try:
            # Load configuration
            self._load_configuration()
            
            # Initialize core components
            self._initialize_display_manager()
            self._initialize_memory_system()
            self._initialize_tool_system()
            self._initialize_model_system()
            self._initialize_agent()
            self._initialize_cli()
            self._initialize_workflow_manager()
            
            print("✅ AI Agent System initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize AI Agent System: {e}")
            sys.exit(1)
    
    def _load_configuration(self) -> None:
        """Load system configuration."""
        # Default configurations
        self.agent_config = AgentConfig(
            model_provider="openai",
            model_name="gpt-3.5-turbo",
            max_reasoning_steps=5,
            tool_timeout=30.0,
            memory_limit=1000
        )
        
        self.display_config = DisplayConfig(
            animation_speed=0.1,
            color_scheme="default",
            progress_style="bar",
            verbose_mode=False
        )
        
        # TODO: Load from config file if provided
        if self.config_path and os.path.exists(self.config_path):
            print(f"📄 Loading configuration from {self.config_path}")
            # Implementation for config file loading would go here
    
    def _initialize_display_manager(self) -> None:
        """Initialize the display management system."""
        from src.display.animation_engine import AnimationEngine
        from src.display.progress_tracker import ProgressTracker
        from src.display.output_formatter import OutputFormatter
        
        # Create display components
        animation_engine = AnimationEngine(color_enabled=True)
        progress_tracker = ProgressTracker(animation_engine=animation_engine, color_enabled=True)
        output_formatter = OutputFormatter(color_enabled=True)
        
        # Create display manager
        display_manager = DisplayManager(color_enabled=True)
        
        self.components['display_manager'] = display_manager
        print("🎨 Display manager initialized")
    
    def _initialize_memory_system(self) -> None:
        """Initialize the memory management system."""
        # Create memory persistence
        memory_dir = Path.home() / ".ai_agent_system" / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        
        memory_persistence = MemoryPersistence(str(memory_dir))
        
        # Create conversation memory
        conversation_memory = ConversationMemory()
        
        self.components['memory'] = conversation_memory
        print("🧠 Memory system initialized")
    
    def _initialize_tool_system(self) -> None:
        """Initialize the tool management system."""
        from src.tools.tool_executor import ToolExecutor
        
        # Create tool registry
        tool_registry = ToolRegistry()
        
        # Register basic tools
        basic_tools = [
            FileOperationsTool(),
            CalculatorTool(),
            WebSearchTool()
        ]
        
        for tool in basic_tools:
            tool_registry.register_tool(tool)
        
        # Create tool executor
        tool_executor = ToolExecutor(
            default_timeout=self.agent_config.tool_timeout
        )
        
        self.components['tool_registry'] = tool_registry
        self.components['tool_executor'] = tool_executor
        print(f"🔧 Tool system initialized with {len(basic_tools)} tools")
    
    def _initialize_model_system(self) -> None:
        """Initialize the AI model management system."""
        # Create AI API manager
        ai_api_manager = AIAPIManager()
        
        # Configure model provider based on config
        try:
            # Try to register and get the provider
            from src.models.data_models import ModelType
            
            if self.agent_config.model_provider == "openai":
                ai_api_manager.register_provider(
                    "main_provider",
                    ModelType.OPENAI,
                    self.agent_config.model_name
                )
            elif self.agent_config.model_provider == "local":
                ai_api_manager.register_provider(
                    "main_provider", 
                    ModelType.LOCAL,
                    self.agent_config.model_name
                )
            
            model_provider = ai_api_manager.get_provider("main_provider")
            self.components['model_provider'] = model_provider
            print(f"🤖 Model system initialized with {self.agent_config.model_provider}")
            
        except Exception as e:
            print(f"⚠️  Model system initialization warning: {e}")
            print("🔄 Falling back to mock model for demonstration")
            
            # Create a simple mock provider
            class MockModelProvider:
                def __init__(self, model_name="mock-model"):
                    self.model_name = model_name
                    from src.models.data_models import ModelInfo, ModelType
                    self.model_info = ModelInfo(
                        name=model_name,
                        model_type=ModelType.LOCAL,
                        description="Mock model for demonstration",
                        max_tokens=1000,
                        supports_streaming=False,
                        api_endpoint="mock://localhost",
                        version="1.0",
                        enabled=True
                    )
                
                def generate_response(self, prompt, context=None):
                    return f"Mock response to: {prompt[:50]}..."
                
                def is_available(self):
                    return True
                
                def get_model_info(self):
                    return self.model_info
                
                def set_parameters(self, params):
                    pass
            
            mock_provider = MockModelProvider("demo-model")
            self.components['model_provider'] = mock_provider
    
    def _initialize_agent(self) -> None:
        """Initialize the ReAct agent."""
        react_agent = ReActAgent(
            model=self.components['model_provider'],
            tool_interface=self.components['tool_registry'],
            memory=self.components['memory'],
            max_iterations=10,
            max_reasoning_steps=self.agent_config.max_reasoning_steps
        )
        
        self.components['agent'] = react_agent
        print("🧠 ReAct agent initialized")
    
    def _initialize_cli(self) -> None:
        """Initialize the CLI application."""
        cli_app = CLIApplication(
            display_manager=self.components['display_manager']
        )
        
        # Inject agent into CLI for task execution
        cli_app.agent = self.components['agent']
        
        self.components['cli'] = cli_app
        print("💻 CLI application initialized")
    
    def _initialize_workflow_manager(self) -> None:
        """Initialize the workflow management system."""
        self.workflow_manager = UserWorkflowManager(
            agent=self.components['agent'],
            memory=self.components['memory'],
            display_manager=self.components['display_manager'],
            tool_registry=self.components['tool_registry']
        )
        
        # Inject workflow manager into CLI
        self.components['cli'].workflow_manager = self.workflow_manager
        print("🔄 Workflow manager initialized")
    
    def run(self) -> None:
        """Run the main application."""
        try:
            print("\n🚀 Starting AI Agent System...")
            print("=" * 50)
            
            # Start the CLI application
            self.components['cli'].start_session()
            
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down gracefully...")
        except Exception as e:
            print(f"\n❌ Application error: {e}")
            sys.exit(1)
        finally:
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up system resources."""
        try:
            # Save memory state
            if 'memory' in self.components:
                self.components['memory'].save_state()
            
            # Clean up other components
            for component_name, component in self.components.items():
                if hasattr(component, 'cleanup'):
                    component.cleanup()
            
            print("🧹 System cleanup completed")
            
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Agent System - ReAct-based intelligent assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start with default configuration
  python main.py --config my.conf  # Start with custom configuration
  python main.py --verbose         # Start with verbose output
  python main.py --model openai    # Use specific model provider
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--model',
        choices=['openai', 'local', 'modelscope'],
        default='openai',
        help='Model provider to use (default: openai)'
    )
    
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )
    
    parser.add_argument(
        '--memory-limit',
        type=int,
        default=1000,
        help='Maximum number of messages to keep in memory (default: 1000)'
    )
    
    return parser


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create and initialize the application
    app = AIAgentSystemApp(config_path=args.config)
    
    # Apply command line overrides
    if hasattr(app, 'agent_config') and app.agent_config:
        if args.model:
            app.agent_config.model_provider = args.model
        if args.memory_limit:
            app.agent_config.memory_limit = args.memory_limit
    
    if hasattr(app, 'display_config') and app.display_config:
        if args.verbose:
            app.display_config.verbose_mode = True
        if args.no_color:
            app.display_config.color_scheme = "none"
    
    # Initialize and run the application
    app.initialize()
    app.run()


if __name__ == "__main__":
    main()