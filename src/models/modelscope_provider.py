"""
ModelScope provider implementation for the AI Agent System.

This module provides integration with ModelScope API, including support for
thinking capabilities and streaming responses.
"""

import os
import asyncio
from typing import Any, Dict, Optional, AsyncGenerator
from ..interfaces.model_interface import ModelInterface
from ..models.data_models import ModelInfo, ModelType
from ..exceptions import ModelError, ConfigurationError


class ModelScopeProvider(ModelInterface):
    """
    ModelScope API provider with thinking capabilities.
    
    This provider integrates with ModelScope's API to provide AI model
    interactions with advanced reasoning capabilities.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-235B-A22B",
        api_key: Optional[str] = None,
        base_url: str = "https://api-inference.modelscope.cn/v1/",
        enable_thinking: bool = True,
        thinking_budget: Optional[int] = None
    ):
        """
        Initialize ModelScope provider.
        
        Args:
            model_name: The ModelScope model ID to use
            api_key: API key for authentication (or from MODELSCOPE_SDK_TOKEN env var)
            base_url: Base URL for ModelScope API
            enable_thinking: Whether to enable thinking capabilities
            thinking_budget: Optional limit on thinking tokens
        """
        self.model_name = model_name
        self.base_url = base_url
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('MODELSCOPE_API_KEY')
        if not self.api_key:
            raise ConfigurationError("ModelScope API key not provided. Set MODELSCOPE_SDK_TOKEN environment variable.")
        
        # Initialize OpenAI client for ModelScope API
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        except ImportError:
            raise ConfigurationError("openai package not installed. Install with: pip install openai")
        
        # Default parameters
        self._parameters = {
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    
    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response from ModelScope API.
        
        Args:
            prompt: The input prompt
            context: Optional context information
            
        Returns:
            Generated response string
            
        Raises:
            ModelError: If response generation fails
        """
        try:
            messages = self._prepare_messages(prompt, context)
            
            extra_body = {
                "enable_thinking": self.enable_thinking
            }
            if self.thinking_budget:
                extra_body["thinking_budget"] = self.thinking_budget
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False,
                extra_body=extra_body,
                **self._parameters
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise ModelError(f"ModelScope API error: {str(e)}")
    
    def generate_reasoning(self, problem: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate reasoning steps for a problem using thinking capabilities.
        
        Args:
            problem: The problem to reason about
            context: Optional context information
            
        Returns:
            Reasoning steps as a string
            
        Raises:
            ModelError: If reasoning generation fails
        """
        reasoning_prompt = f"""
        Please think through this problem step by step and provide your reasoning:
        
        Problem: {problem}
        
        Think carefully about:
        1. What information do I have?
        2. What do I need to find out?
        3. What steps should I take?
        4. What are the potential challenges?
        5. How can I verify my solution?
        """
        
        # Force thinking mode for reasoning
        original_thinking = self.enable_thinking
        self.enable_thinking = True
        
        try:
            result = self.generate_response(reasoning_prompt, context)
            return result
        finally:
            self.enable_thinking = original_thinking
    
    def is_available(self) -> bool:
        """
        Check if ModelScope API is available.
        
        Returns:
            True if available, False otherwise
        """
        try:
            # Simple test call to check availability
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                stream=False,
                extra_body={"enable_thinking": False}
            )
            return True
        except Exception:
            return False
    
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the ModelScope model.
        
        Returns:
            ModelInfo object with model details
        """
        return ModelInfo(
            name=self.model_name,
            model_type=ModelType.CLOUD,
            max_tokens=self._parameters.get("max_tokens", 2048),
            supports_streaming=True,
            supports_reasoning=True,
            provider="ModelScope",
            capabilities=["text_generation", "reasoning", "thinking", "streaming"]
        )
    
    def validate_input(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate input before sending to ModelScope.
        
        Args:
            prompt: Input prompt to validate
            context: Optional context to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not prompt or not isinstance(prompt, str):
            return False
        
        if len(prompt.strip()) == 0:
            return False
        
        # Check token estimate (rough)
        estimated_tokens = self.estimate_tokens(prompt)
        if estimated_tokens > self._parameters.get("max_tokens", 2048):
            return False
        
        return True
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token for Chinese/English mixed content
        return len(text) // 4
    
    def supports_streaming(self) -> bool:
        """
        Check if streaming is supported.
        
        Returns:
            True (ModelScope supports streaming)
        """
        return True
    
    async def generate_response_stream(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from ModelScope.
        
        Args:
            prompt: Input prompt
            context: Optional context information
            
        Yields:
            Response chunks as they arrive
            
        Raises:
            ModelError: If streaming fails
        """
        try:
            messages = self._prepare_messages(prompt, context)
            
            extra_body = {
                "enable_thinking": self.enable_thinking
            }
            if self.thinking_budget:
                extra_body["thinking_budget"] = self.thinking_budget
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                extra_body=extra_body,
                **self._parameters
            )
            
            thinking_complete = False
            
            for chunk in response:
                # Handle thinking content
                thinking_chunk = getattr(chunk.choices[0].delta, 'reasoning_content', '')
                if thinking_chunk:
                    if not thinking_complete:
                        yield f"[THINKING] {thinking_chunk}"
                
                # Handle regular content
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    if not thinking_complete:
                        yield "\n[RESPONSE] "
                        thinking_complete = True
                    yield content_chunk
                    
        except Exception as e:
            raise ModelError(f"ModelScope streaming error: {str(e)}")
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set model parameters.
        
        Args:
            parameters: Dictionary of parameters to set
            
        Raises:
            ValueError: If parameters are invalid
        """
        valid_params = {
            "temperature", "max_tokens", "top_p", 
            "frequency_penalty", "presence_penalty"
        }
        
        for key, value in parameters.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter: {key}")
            
            # Validate parameter values
            if key == "temperature" and not (0.0 <= value <= 2.0):
                raise ValueError("Temperature must be between 0.0 and 2.0")
            elif key == "max_tokens" and not (1 <= value <= 4096):
                raise ValueError("Max tokens must be between 1 and 4096")
            elif key in ["top_p"] and not (0.0 <= value <= 1.0):
                raise ValueError(f"{key} must be between 0.0 and 1.0")
            
            self._parameters[key] = value
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current model parameters.
        
        Returns:
            Dictionary of current parameters
        """
        return self._parameters.copy()
    
    def reset_parameters(self) -> None:
        """Reset parameters to defaults."""
        self._parameters = {
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    
    def _prepare_messages(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> list:
        """
        Prepare messages for API call.
        
        Args:
            prompt: User prompt
            context: Optional context information
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system message if context provided
        if context and "system_message" in context:
            messages.append({
                "role": "system",
                "content": context["system_message"]
            })
        
        # Add conversation history if provided
        if context and "history" in context:
            for msg in context["history"]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    def get_thinking_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Get response with separate thinking and answer components.
        
        Args:
            prompt: Input prompt
            context: Optional context information
            
        Returns:
            Dictionary with 'thinking' and 'answer' keys
            
        Raises:
            ModelError: If response generation fails
        """
        if not self.enable_thinking:
            # If thinking disabled, just return regular response
            response = self.generate_response(prompt, context)
            return {"thinking": "", "answer": response}
        
        try:
            messages = self._prepare_messages(prompt, context)
            
            extra_body = {"enable_thinking": True}
            if self.thinking_budget:
                extra_body["thinking_budget"] = self.thinking_budget
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                extra_body=extra_body,
                **self._parameters
            )
            
            thinking_content = ""
            answer_content = ""
            thinking_complete = False
            
            for chunk in response:
                thinking_chunk = getattr(chunk.choices[0].delta, 'reasoning_content', '')
                if thinking_chunk:
                    thinking_content += thinking_chunk
                
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    thinking_complete = True
                    answer_content += content_chunk
            
            return {
                "thinking": thinking_content,
                "answer": answer_content
            }
            
        except Exception as e:
            raise ModelError(f"ModelScope thinking response error: {str(e)}")