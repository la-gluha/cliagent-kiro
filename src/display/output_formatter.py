"""
Output formatter for the AI Agent System.

This module provides formatting capabilities for different types of content
with support for colors, styles, and structured output.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from ..interfaces.display_interface import OutputStyle
from ..models.data_models import Message, TaskResult, ToolInfo, AgentState


class ColorCodes:
    """ANSI color codes for terminal output."""
    
    # Reset
    RESET = "\033[0m"
    
    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    STRIKETHROUGH = "\033[9m"


class OutputFormatter:
    """
    Handles formatting of various content types for display.
    
    Provides methods to format messages, data structures, and other content
    with appropriate styling and colors.
    """
    
    def __init__(self, color_enabled: bool = True):
        """
        Initialize the output formatter.
        
        Args:
            color_enabled: Whether to enable color output
        """
        self.color_enabled = color_enabled
        self._style_map = {
            OutputStyle.NORMAL: "",
            OutputStyle.SUCCESS: ColorCodes.BRIGHT_GREEN,
            OutputStyle.ERROR: ColorCodes.BRIGHT_RED,
            OutputStyle.WARNING: ColorCodes.BRIGHT_YELLOW,
            OutputStyle.INFO: ColorCodes.BRIGHT_BLUE,
            OutputStyle.HIGHLIGHT: ColorCodes.BRIGHT_CYAN,
            OutputStyle.MUTED: ColorCodes.DIM
        }
    
    def set_color_enabled(self, enabled: bool) -> None:
        """
        Enable or disable color output.
        
        Args:
            enabled: Whether to enable color output
        """
        self.color_enabled = enabled
    
    def apply_style(self, text: str, style: OutputStyle) -> str:
        """
        Apply a style to text.
        
        Args:
            text: The text to style
            style: The style to apply
            
        Returns:
            Styled text
        """
        if not self.color_enabled:
            return text
        
        color_code = self._style_map.get(style, "")
        if color_code:
            return f"{color_code}{text}{ColorCodes.RESET}"
        return text
    
    def format_message(self, message: Message) -> str:
        """
        Format a message for display.
        
        Args:
            message: The message to format
            
        Returns:
            Formatted message string
        """
        timestamp = message.timestamp.strftime("%H:%M:%S")
        role_colors = {
            "user": ColorCodes.BRIGHT_CYAN,
            "agent": ColorCodes.BRIGHT_GREEN,
            "system": ColorCodes.BRIGHT_YELLOW
        }
        
        role_color = role_colors.get(message.role.value, "")
        role_text = message.role.value.upper()
        
        if self.color_enabled and role_color:
            role_display = f"{role_color}{role_text}{ColorCodes.RESET}"
        else:
            role_display = role_text
        
        # Format the message content with proper indentation
        content_lines = message.content.split('\n')
        formatted_content = '\n'.join(f"  {line}" if line.strip() else "" for line in content_lines)
        
        return f"[{timestamp}] {role_display}:\n{formatted_content}"
    
    def format_task_result(self, result: TaskResult) -> str:
        """
        Format a task result for display.
        
        Args:
            result: The task result to format
            
        Returns:
            Formatted task result string
        """
        status = "SUCCESS" if result.success else "FAILED"
        status_style = OutputStyle.SUCCESS if result.success else OutputStyle.ERROR
        status_text = self.apply_style(status, status_style)
        
        lines = [f"Task Result: {status_text}"]
        
        if result.execution_time > 0:
            time_text = f"Execution Time: {result.execution_time:.2f}s"
            lines.append(self.apply_style(time_text, OutputStyle.MUTED))
        
        if result.error_message:
            error_text = f"Error: {result.error_message}"
            lines.append(self.apply_style(error_text, OutputStyle.ERROR))
        
        if result.steps_taken:
            lines.append(self.apply_style("Steps taken:", OutputStyle.INFO))
            for i, step in enumerate(result.steps_taken, 1):
                lines.append(f"  {i}. {step}")
        
        if result.result is not None:
            lines.append(self.apply_style("Result:", OutputStyle.INFO))
            result_str = self._format_any_content(result.result, indent=2)
            lines.append(result_str)
        
        return '\n'.join(lines)
    
    def format_tool_info(self, tool_info: ToolInfo) -> str:
        """
        Format tool information for display.
        
        Args:
            tool_info: The tool info to format
            
        Returns:
            Formatted tool info string
        """
        name = self.apply_style(tool_info.name, OutputStyle.HIGHLIGHT)
        status = "ENABLED" if tool_info.enabled else "DISABLED"
        status_style = OutputStyle.SUCCESS if tool_info.enabled else OutputStyle.MUTED
        status_text = self.apply_style(status, status_style)
        
        lines = [f"{name} ({status_text})"]
        lines.append(f"  Category: {tool_info.category.value}")
        lines.append(f"  Description: {tool_info.description}")
        
        if tool_info.parameters:
            lines.append("  Parameters:")
            for param, details in tool_info.parameters.items():
                param_text = self.apply_style(param, OutputStyle.INFO)
                if isinstance(details, dict):
                    param_type = details.get('type', 'unknown')
                    required = details.get('required', False)
                    req_text = " (required)" if required else " (optional)"
                    lines.append(f"    {param_text}: {param_type}{req_text}")
                else:
                    lines.append(f"    {param_text}: {details}")
        
        return '\n'.join(lines)
    
    def format_agent_state(self, state: AgentState) -> str:
        """
        Format agent state for display.
        
        Args:
            state: The agent state to format
            
        Returns:
            Formatted agent state string
        """
        status = "ACTIVE" if state.is_active else "INACTIVE"
        status_style = OutputStyle.SUCCESS if state.is_active else OutputStyle.MUTED
        status_text = self.apply_style(status, status_style)
        
        lines = [f"Agent Status: {status_text}"]
        
        if state.current_task:
            task_text = self.apply_style(f"Current Task: {state.current_task}", OutputStyle.INFO)
            lines.append(task_text)
        
        if state.reasoning_history:
            lines.append(self.apply_style("Recent Reasoning:", OutputStyle.INFO))
            # Show last 3 reasoning steps
            recent_steps = state.reasoning_history[-3:]
            for i, step in enumerate(recent_steps, 1):
                lines.append(f"  {i}. {step}")
        
        if state.action_history:
            lines.append(self.apply_style(f"Actions Taken: {len(state.action_history)}", OutputStyle.INFO))
        
        return '\n'.join(lines)
    
    def format_json(self, data: Any, indent: int = 2) -> str:
        """
        Format data as JSON with syntax highlighting.
        
        Args:
            data: The data to format as JSON
            indent: Indentation level
            
        Returns:
            Formatted JSON string
        """
        try:
            json_str = json.dumps(data, indent=indent, default=str, ensure_ascii=False)
            if self.color_enabled:
                # Simple syntax highlighting for JSON
                json_str = json_str.replace('":', f'"{ColorCodes.CYAN}:{ColorCodes.RESET}')
                json_str = json_str.replace('true', f'{ColorCodes.GREEN}true{ColorCodes.RESET}')
                json_str = json_str.replace('false', f'{ColorCodes.RED}false{ColorCodes.RESET}')
                json_str = json_str.replace('null', f'{ColorCodes.YELLOW}null{ColorCodes.RESET}')
            return json_str
        except (TypeError, ValueError) as e:
            return f"Error formatting JSON: {e}"
    
    def format_list(self, items: List[Any], numbered: bool = True) -> str:
        """
        Format a list of items for display.
        
        Args:
            items: The list of items to format
            numbered: Whether to number the items
            
        Returns:
            Formatted list string
        """
        if not items:
            return self.apply_style("(empty list)", OutputStyle.MUTED)
        
        lines = []
        for i, item in enumerate(items, 1):
            prefix = f"{i}. " if numbered else "• "
            item_str = self._format_any_content(item)
            lines.append(f"{prefix}{item_str}")
        
        return '\n'.join(lines)
    
    def format_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """
        Format data as a simple table.
        
        Args:
            headers: Column headers
            rows: Table rows
            
        Returns:
            Formatted table string
        """
        if not headers or not rows:
            return self.apply_style("(empty table)", OutputStyle.MUTED)
        
        # Calculate column widths
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Format header
        header_line = " | ".join(header.ljust(width) for header, width in zip(headers, col_widths))
        separator = "-+-".join("-" * width for width in col_widths)
        
        lines = [
            self.apply_style(header_line, OutputStyle.HIGHLIGHT),
            separator
        ]
        
        # Format rows
        for row in rows:
            row_line = " | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths))
            lines.append(row_line)
        
        return '\n'.join(lines)
    
    def _format_any_content(self, content: Any, indent: int = 0) -> str:
        """
        Format any content type for display.
        
        Args:
            content: The content to format
            indent: Indentation level
            
        Returns:
            Formatted content string
        """
        indent_str = "  " * indent
        
        if isinstance(content, str):
            return content
        elif isinstance(content, (int, float, bool)):
            return str(content)
        elif isinstance(content, dict):
            if not content:
                return "{}"
            lines = ["{"]
            for key, value in content.items():
                value_str = self._format_any_content(value, indent + 1)
                lines.append(f"{indent_str}  {key}: {value_str}")
            lines.append(f"{indent_str}}}")
            return '\n'.join(lines)
        elif isinstance(content, list):
            if not content:
                return "[]"
            lines = ["["]
            for item in content:
                item_str = self._format_any_content(item, indent + 1)
                lines.append(f"{indent_str}  {item_str}")
            lines.append(f"{indent_str}]")
            return '\n'.join(lines)
        else:
            return str(content)
    
    def create_separator(self, char: str = "-", length: int = 50) -> str:
        """
        Create a separator line.
        
        Args:
            char: Character to use for the separator
            length: Length of the separator
            
        Returns:
            Separator string
        """
        return self.apply_style(char * length, OutputStyle.MUTED)
    
    def create_header(self, title: str, char: str = "=") -> str:
        """
        Create a header with title.
        
        Args:
            title: The header title
            char: Character to use for decoration
            
        Returns:
            Formatted header string
        """
        title_line = self.apply_style(title, OutputStyle.HIGHLIGHT)
        separator = self.apply_style(char * len(title), OutputStyle.MUTED)
        return f"{separator}\n{title_line}\n{separator}"