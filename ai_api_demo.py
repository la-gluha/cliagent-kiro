#!/usr/bin/env python3
"""
AI API Demo - Comprehensive demonstration of different AI API integrations.

This demo shows how to use various AI APIs including:
1. ModelScope API (with thinking capabilities)
2. OpenAI API (standard and streaming)
3. Our custom model providers (OpenAI and Local)
4. Error handling and best practices

Usage:
    python ai_api_demo.py [--provider modelscope|openai|local] [--stream]
"""

import os
import sys
import asyncio
import argparse
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.openai_provider import OpenAIProvider
from src.models.local_model_provider import LocalModelProvider
from src.models.mock_provider import MockModelProvider
from src.models.data_models import ModelInfo, ModelType
from src.exceptions import ModelError, ConfigurationError


class ModelScopeDemo:
    """Demo for ModelScope API with thinking capabilities."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize ModelScope demo."""
        try:
            from openai import OpenAI
        except ImportError:
            print("Error: openai package not installed. Install with: pip install openai")
            sys.exit(1)
        
        self.api_key = api_key or os.getenv('MODELSCOPE_SDK_TOKEN')
        if not self.api_key:
            print("Warning: No ModelScope API key found. Set MODELSCOPE_SDK_TOKEN environment variable.")
            return
        
        self.client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1/',
            api_key=self.api_key
        )
    
    def demo_thinking_response(self, question: str = "9.9和9.11谁大"):
        """Demonstrate ModelScope API with thinking capabilities."""
        if not self.api_key:
            print("Skipping ModelScope demo - no API key provided")
            return
        
        print("=== ModelScope API Demo (with Thinking) ===")
        print(f"Question: {question}")
        print("\n--- Thinking Process ---")
        
        # Set extra_body for thinking control
        extra_body = {
            "enable_thinking": True,
            # "thinking_budget": 4096  # Optional: control thinking tokens
        }
        
        try:
            response = self.client.chat.completions.create(
                model='Qwen/Qwen3-235B-A22B',  # ModelScope Model-Id
                messages=[
                    {
                        'role': 'user',
                        'content': question
                    }
                ],
                stream=True,
                extra_body=extra_body
            )
            
            done_thinking = False
            for chunk in response:
                thinking_chunk = chunk.choices[0].delta.reasoning_content
                answer_chunk = chunk.choices[0].delta.content
                
                if thinking_chunk != '':
                    print(thinking_chunk, end='', flush=True)
                elif answer_chunk != '':
                    if not done_thinking:
                        print('\n\n=== Final Answer ===\n')
                        done_thinking = True
                    print(answer_chunk, end='', flush=True)
            
            print("\n\n=== ModelScope Demo Complete ===\n")
            
        except Exception as e:
            print(f"Error calling ModelScope API: {e}")
    
    def demo_standard_response(self, question: str = "What is the capital of France?"):
        """Demonstrate standard ModelScope API call without thinking."""
        if not self.api_key:
            print("Skipping ModelScope standard demo - no API key provided")
            return
        
        print("=== ModelScope API Demo (Standard) ===")
        print(f"Question: {question}")
        
        try:
            response = self.client.chat.completions.create(
                model='Qwen/Qwen3-235B-A22B',
                messages=[
                    {
                        'role': 'user',
                        'content': question
                    }
                ],
                stream=False,
                extra_body={"enable_thinking": False}
            )
            
            answer = response.choices[0].message.content
            print(f"Answer: {answer}")
            print("\n=== ModelScope Standard Demo Complete ===\n")
            
        except Exception as e:
            print(f"Error calling ModelScope API: {e}")


class OpenAIDemo:
    """Demo for OpenAI API using our custom provider."""
    
    def __init__(self):
        """Initialize OpenAI demo."""
        self.provider = None
        try:
            self.provider = OpenAIProvider(
                model_name="gpt-3.5-turbo",
                api_key=os.getenv('OPENAI_API_KEY')
            )
        except (ConfigurationError, ModelError) as e:
            print(f"OpenAI provider initialization failed: {e}")
    
    def demo_standard_response(self, question: str = "Explain quantum computing in simple terms"):
        """Demonstrate standard OpenAI API call."""
        if not self.provider:
            print("Skipping OpenAI demo - provider not available")
            return
        
        print("=== OpenAI API Demo (Standard) ===")
        print(f"Question: {question}")
        
        try:
            response = self.provider.generate_response(question)
            print(f"Answer: {response}")
            print("\n=== OpenAI Standard Demo Complete ===\n")
            
        except ModelError as e:
            print(f"Error calling OpenAI API: {e}")
    
    def demo_reasoning_response(self, problem: str = "How would you solve world hunger?"):
        """Demonstrate OpenAI reasoning capabilities."""
        if not self.provider:
            print("Skipping OpenAI reasoning demo - provider not available")
            return
        
        print("=== OpenAI API Demo (Reasoning) ===")
        print(f"Problem: {problem}")
        
        try:
            reasoning = self.provider.generate_reasoning(problem)
            print(f"Reasoning: {reasoning}")
            print("\n=== OpenAI Reasoning Demo Complete ===\n")
            
        except ModelError as e:
            print(f"Error calling OpenAI API: {e}")
    
    async def demo_streaming_response(self, question: str = "Tell me a short story about AI"):
        """Demonstrate OpenAI streaming capabilities."""
        if not self.provider:
            print("Skipping OpenAI streaming demo - provider not available")
            return
        
        print("=== OpenAI API Demo (Streaming) ===")
        print(f"Question: {question}")
        print("Response: ", end='', flush=True)
        
        try:
            async for chunk in self.provider.generate_response_stream(question):
                print(chunk, end='', flush=True)
            
            print("\n\n=== OpenAI Streaming Demo Complete ===\n")
            
        except (ModelError, NotImplementedError) as e:
            print(f"Error with OpenAI streaming: {e}")


class LocalModelDemo:
    """Demo for local model provider."""
    
    def __init__(self):
        """Initialize local model demo."""
        self.provider = None
        try:
            # Try to initialize with Ollama first
            self.provider = LocalModelProvider(
                model_name="demo-local-model",
                backend="ollama"
            )
        except (ModelError, ConfigurationError) as e:
            print(f"Failed to initialize local model provider: {e}")
            print("Note: This demo requires Ollama to be running on localhost:11434")
            print("Install Ollama from https://ollama.ai/ and run 'ollama serve' to enable local model demos")
    
    def demo_mock_responses(self):
        """Demonstrate local model mock responses."""
        if not self.provider:
            print("Skipping Local Model demo - provider not available")
            return
            
        print("=== Local Model Demo ===")
        
        test_prompts = [
            "Hello, how are you?",
            "What is machine learning?",
            "Think step by step: How do you make coffee?",
            "Explain the concept of recursion"
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            try:
                response = self.provider.generate_response(prompt)
                print(f"Response: {response}")
            except ModelError as e:
                print(f"Error: {e}")
        
        print("\n=== Local Model Demo Complete ===\n")
    
    async def demo_streaming_mock(self, question: str = "Explain artificial intelligence"):
        """Demonstrate local model streaming."""
        if not self.provider:
            print("Skipping Local Model streaming demo - provider not available")
            return
            
        print("=== Local Model Demo (Streaming) ===")
        print(f"Question: {question}")
        print("Response: ", end='', flush=True)
        
        try:
            async for chunk in self.provider.generate_response_stream(question):
                print(chunk, end='', flush=True)
            
            print("\n\n=== Local Model Streaming Demo Complete ===\n")
            
        except (ModelError, NotImplementedError) as e:
            print(f"Error with local model streaming: {e}")
    
    def demo_model_management(self):
        """Demonstrate model management features."""
        if not self.provider:
            print("Skipping Local Model management demo - provider not available")
            return
            
        print("=== Local Model Management Demo ===")
        
        try:
            # Show model status
            status = self.provider.get_model_status()
            print(f"Model Status: {status}")
            
            # Show memory usage
            memory = self.provider.get_memory_usage()
            print(f"Memory Usage: {memory}")
            
            # Show parameters
            params = self.provider.get_parameters()
            print(f"Parameters: {params}")
            
            # Test parameter changes
            print("\nChanging parameters...")
            self.provider.set_parameters({"temperature": 0.9, "max_tokens": 500})
            new_params = self.provider.get_parameters()
            print(f"New Parameters: {new_params}")
            
            # Reset parameters
            print("\nResetting parameters...")
            self.provider.reset_parameters()
            reset_params = self.provider.get_parameters()
            print(f"Reset Parameters: {reset_params}")
            
        except (ModelError, AttributeError) as e:
            print(f"Error in model management demo: {e}")
        
        print("\n=== Local Model Management Demo Complete ===\n")


class MockDemo:
    """Demo for mock AI provider (works without external dependencies)."""
    
    def __init__(self):
        """Initialize mock demo."""
        self.provider = MockModelProvider(
            model_name="demo-mock-ai",
            response_delay=0.3
        )
    
    def demo_mock_responses(self):
        """Demonstrate mock AI responses."""
        print("=== Mock AI Demo ===")
        print("This demo works without any API keys or external services!")
        
        test_prompts = [
            "Hello, how are you?",
            "What is machine learning?",
            "Explain how to write Python code",
            "Think step by step: How do you make coffee?",
            "What is the meaning of life?"
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            try:
                response = self.provider.generate_response(prompt)
                print(f"Response: {response}")
            except ModelError as e:
                print(f"Error: {e}")
        
        print("\n=== Mock AI Demo Complete ===\n")
    
    async def demo_streaming_mock(self, question: str = "Explain artificial intelligence"):
        """Demonstrate mock streaming responses."""
        print("=== Mock AI Demo (Streaming) ===")
        print(f"Question: {question}")
        print("Response: ", end='', flush=True)
        
        try:
            async for chunk in self.provider.generate_response_stream(question):
                print(chunk, end='', flush=True)
            
            print("\n\n=== Mock AI Streaming Demo Complete ===\n")
            
        except (ModelError, NotImplementedError) as e:
            print(f"Error with mock streaming: {e}")
    
    def demo_reasoning(self, problem: str = "How can we solve climate change?"):
        """Demonstrate mock reasoning capabilities."""
        print("=== Mock AI Demo (Reasoning) ===")
        print(f"Problem: {problem}")
        
        try:
            reasoning = self.provider.generate_reasoning(problem)
            print(f"Reasoning: {reasoning}")
            print("\n=== Mock AI Reasoning Demo Complete ===\n")
            
        except ModelError as e:
            print(f"Error with mock reasoning: {e}")
    
    def demo_model_management(self):
        """Demonstrate mock model management features."""
        print("=== Mock AI Model Management Demo ===")
        
        try:
            # Show model status
            status = self.provider.get_model_status()
            print(f"Model Status: {status}")
            
            # Show memory usage
            memory = self.provider.get_memory_usage()
            print(f"Memory Usage: {memory}")
            
            # Show parameters
            params = self.provider.get_parameters()
            print(f"Parameters: {params}")
            
            # Test parameter changes
            print("\nChanging parameters...")
            self.provider.set_parameters({"temperature": 0.9, "response_delay": 0.1})
            new_params = self.provider.get_parameters()
            print(f"New Parameters: {new_params}")
            
            # Reset parameters
            print("\nResetting parameters...")
            self.provider.reset_parameters()
            reset_params = self.provider.get_parameters()
            print(f"Reset Parameters: {reset_params}")
            
        except (ModelError, AttributeError) as e:
            print(f"Error in mock model management demo: {e}")
        
        print("\n=== Mock AI Model Management Demo Complete ===\n")


class ComparisonDemo:
    """Demo comparing different providers."""
    
    def __init__(self):
        """Initialize comparison demo."""
        self.providers = {}
        
        # Always include mock provider for demonstration
        self.providers['mock'] = MockModelProvider(
            model_name="mock-ai-comparison",
            response_delay=0.2
        )
        
        # Initialize available providers
        try:
            self.providers['openai'] = OpenAIProvider(
                model_name="gpt-3.5-turbo",
                api_key=os.getenv('OPENAI_API_KEY')
            )
        except (ConfigurationError, ModelError):
            print("OpenAI provider not available for comparison")
        
        try:
            self.providers['local'] = LocalModelProvider(
                model_name="comparison-local-model",
                backend="ollama"
            )
        except (ConfigurationError, ModelError):
            print("Local provider not available for comparison")
    
    def demo_provider_comparison(self, question: str = "What is the meaning of life?"):
        """Compare responses from different providers."""
        print("=== Provider Comparison Demo ===")
        print(f"Question: {question}")
        
        for name, provider in self.providers.items():
            print(f"\n--- {name.upper()} Provider ---")
            try:
                response = provider.generate_response(question)
                print(f"Response: {response}")
                
                # Show provider info
                info = provider.get_model_info()
                print(f"Model: {info.name} ({info.model_type.value})")
                print(f"Max Tokens: {info.max_tokens}")
                print(f"Supports Streaming: {info.supports_streaming}")
                
            except ModelError as e:
                print(f"Error: {e}")
        
        print("\n=== Provider Comparison Complete ===\n")


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='AI API Demo')
    parser.add_argument('--provider', choices=['modelscope', 'openai', 'local', 'all'], 
                       default='all', help='Which provider to demo')
    parser.add_argument('--stream', action='store_true', help='Include streaming demos')
    parser.add_argument('--question', type=str, help='Custom question to ask')
    
    args = parser.parse_args()
    
    print("🤖 AI API Demo - Comprehensive AI Integration Examples")
    print("=" * 60)
    
    # Custom question for demos
    question = args.question or "What is artificial intelligence?"
    
    # ModelScope Demo
    if args.provider in ['modelscope', 'all']:
        modelscope_demo = ModelScopeDemo()
        modelscope_demo.demo_thinking_response("9.9和9.11谁大")
        modelscope_demo.demo_standard_response(question)
    
    # OpenAI Demo
    if args.provider in ['openai', 'all']:
        openai_demo = OpenAIDemo()
        openai_demo.demo_standard_response(question)
        openai_demo.demo_reasoning_response("How can we solve climate change?")
        
        if args.stream:
            await openai_demo.demo_streaming_response(question)
    
    # Local Model Demo
    if args.provider in ['local', 'all']:
        local_demo = LocalModelDemo()
        local_demo.demo_mock_responses()
        local_demo.demo_model_management()
        
        if args.stream:
            await local_demo.demo_streaming_mock(question)
    
    # Comparison Demo
    if args.provider == 'all':
        comparison_demo = ComparisonDemo()
        comparison_demo.demo_provider_comparison(question)
    
    print("🎉 Demo Complete!")
    print("\nTo run specific demos:")
    print("  python ai_api_demo.py --provider modelscope")
    print("  python ai_api_demo.py --provider openai --stream")
    print("  python ai_api_demo.py --provider local")
    print("  python ai_api_demo.py --question 'Your custom question here'")
    print("\nEnvironment variables needed:")
    print("  MODELSCOPE_SDK_TOKEN - for ModelScope API")
    print("  OPENAI_API_KEY - for OpenAI API")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())