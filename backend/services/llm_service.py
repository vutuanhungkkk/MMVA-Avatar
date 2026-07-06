from typing import Any, Dict, List, Optional
import threading

from assistant.config import ENABLE_TOOL_CALLING, SIMPLE_TOOLS, SYSTEM_MESSAGE
from assistant.providers.llm import get_llm_provider, LLMProvider
from assistant.context import EnhancedConversationContext, ContextProviderRegistry, MCPContextProvider
from assistant.tools import ToolLoop, ToolRegistry, extract_clipboard_text_tool
from assistant.utils import log

class LLMService:
    def __init__(self):
        self.llm_provider: LLMProvider = get_llm_provider()
        
        self.conversation_context = EnhancedConversationContext()
        self.context_provider_registry = ContextProviderRegistry()
        self.context_provider_registry.register(MCPContextProvider())
        
        self.tool_registry = ToolRegistry()
        self._register_builtin_tools()
        
        self.tool_loop = ToolLoop(
            llm_provider=self.llm_provider,
            tool_registry=self.tool_registry,
            tools_enabled=ENABLE_TOOL_CALLING,
        )
        
        if not ENABLE_TOOL_CALLING:
            log("Tool calling disabled via ASSISTANT_DISABLE_TOOLS.", title="TOOLS", style="bold yellow")
        if SIMPLE_TOOLS:
            log("Simple tools mode enabled. Vision tools disabled.", title="TOOLS", style="bold blue")

    def _register_builtin_tools(self) -> None:
        """Register built-in tools with the registry."""
        self.tool_registry.register(
            name="extract_clipboard_text",
            description="Extract the latest textual content from the user's clipboard.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: extract_clipboard_text_tool(),
        )
        
    def register_tool(self, name: str, description: str, parameters: dict, handler: Any):
        self.tool_registry.register(name=name, description=description, parameters=parameters, handler=handler)

    def generate_text_response(self, system_prompt: str, user_prompt: str) -> str:
        """Run a short one-off LLM prompt and return the full text response."""
        convo = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = ""
        try:
            for chunk in self.tool_loop.stream(convo):
                response += chunk
        except Exception as exc:
            log(f"Internal LLM helper failed: {exc}", title="LLM", style="bold yellow")
        return response.strip()

    def stream_response(self, messages: List[Dict[str, Any]]):
        """Stream response chunks from the tool loop."""
        for chunk in self.tool_loop.stream(messages):
            yield chunk

    def gather_context(self, prompt: str) -> str:
        """Gather context from registered providers."""
        return self.context_provider_registry.gather(
            prompt=prompt, conversation_history=self.conversation_context.history
        )
