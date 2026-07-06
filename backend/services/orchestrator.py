import io
import threading
from typing import Any, Dict, List, Optional
from pathlib import Path

from assistant.config import SYSTEM_MESSAGE
from assistant.utils import console, log, save_log
import importlib

Panel = importlib.import_module("rich.panel").Panel

from backend.services.llm_service import LLMService
from backend.services.rag_service import RAGService
from backend.services.voice_service import VoiceService

class Orchestrator:
    def __init__(self):
        self.llm_service = LLMService()
        self.rag_service = RAGService()
        self.voice_service = VoiceService()

        self._summary_lock = threading.Lock()
        self._summary_text = ""
        self._summarizing = False
        self._recent_turn_limit = 6

        self.convo: List[Dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]

        # Register search knowledge base tool into LLM service using RAG service
        self.llm_service.register_tool(
            name="search_knowledge_base",
            description="Search uploaded local documents and return the most relevant passages.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Question or search text."},
                    "top_k": {"type": "integer", "description": "Number of candidate chunks.", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
            handler=lambda query, top_k=10: self.rag_service.search_knowledge_base(query=query, top_k=top_k)
        )

    def _system_prompt(self) -> str:
        return (
            f"{SYSTEM_MESSAGE}\n\n"
            "You are a medical agentic assistant with access to tools.\n"
            "Use the search_knowledge_base tool when the user asks about medications, symptoms, "
            "lab results, first-aid procedures, or any health-related topic that may be covered "
            "in uploaded medical documents.\n"
            "Answer directly when no tool is needed.\n"
            "If a question is ambiguous or refers to prior context, rewrite it mentally before deciding on tools.\n"
            "You may call multiple tools in sequence and then synthesize the final answer.\n"
            "Do not expose hidden chain-of-thought; provide concise answers with useful evidence."
        )

    def _summarize_history(self) -> None:
        if len(self.convo) <= (self._recent_turn_limit * 2) + 1:
            return
        with self._summary_lock:
            if self._summarizing or len(self.convo) <= (self._recent_turn_limit * 2) + 1:
                return
            self._summarizing = True

        def _worker() -> None:
            try:
                with self._summary_lock:
                    historical_messages = self.convo[1 : max(1, len(self.convo) - (self._recent_turn_limit * 2))]
                if not historical_messages:
                    return
                transcript_lines = []
                for message in historical_messages:
                    role = message.get("role", "unknown")
                    content = message.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
                    transcript_lines.append(f"{role}: {content}")

                summary_prompt = "Summarize the conversation below into 6-10 concise bullet points.\n\n" + "\n".join(transcript_lines)
                new_summary = self.llm_service.generate_text_response("You summarize conversations for long-term memory.", summary_prompt)
                if not new_summary:
                    return

                with self._summary_lock:
                    if self._summary_text:
                        self._summary_text = f"{self._summary_text}\n{new_summary}"
                    else:
                        self._summary_text = new_summary
                    self.convo = [self.convo[0]] + self.convo[-(self._recent_turn_limit * 2):]
            finally:
                with self._summary_lock:
                    self._summarizing = False

        threading.Thread(target=_worker, daemon=True).start()

    def _build_user_message(self, prompt: str, img_context: Optional[str] = None) -> Dict[str, Any]:
        if img_context and self.llm_service.llm_provider.supports_vision:
            medical_vision_hint = (
                "\n\n[MEDICAL IMAGE ANALYSIS INSTRUCTIONS]: "
                "If this image contains a prescription, carefully identify ALL medication names, dosages, "
                "and frequencies — including handwritten text. "
                "If it is a lab report, extract key values and flag any abnormal results. "
                "If it shows a skin condition or wound, describe the visible characteristics objectively. "
                "Present findings in a structured, easy-to-read format."
            )
            enhanced_prompt = (prompt + medical_vision_hint) if prompt.strip() else (
                "Please analyze this medical image." + medical_vision_hint
            )
            return {"role": "user", "content": [
                {"type": "text", "text": enhanced_prompt},
                {"type": "image_url", "image_url": {"url": img_context}}
            ]}
        if img_context and not self.llm_service.llm_provider.supports_vision:
            prompt = f"{prompt}\n\n[User attached an image but vision is not supported]"
        return {"role": "user", "content": prompt}

    def _build_message_payload(self, prompt: str, img_context: Optional[str] = None, provider_context: Optional[str] = None, rag_context: Optional[str] = None) -> List[Dict[str, Any]]:
        messages = [{"role": "system", "content": self._system_prompt()}]
        if self._summary_text.strip():
            messages.append({"role": "system", "content": f"Conversation summary so far:\n{self._summary_text.strip()}"})
        if provider_context:
            messages.append({"role": "system", "content": f"Additional provider context:\n{provider_context}"})
        if rag_context and rag_context.strip():
            messages.append({
                "role": "system",
                "content": (
                    "CRITICAL INSTRUCTION: The following information is extracted directly from the user's document. "
                    "You MUST use this as your absolute source of truth to answer their question. "
                    "Maintain your professional medical assistant persona and ALWAYS include your medical disclaimer if providing medical advice. "
                    "DO NOT use phrases like 'According to the document' or 'Based on the provided text'. Just integrate the information naturally into your answer.\n\n"
                    f"{rag_context}"
                )
            })

        recent_history = self.convo[1:]
        if len(recent_history) > self._recent_turn_limit * 2:
            recent_history = recent_history[-(self._recent_turn_limit * 2):]
        messages.extend(recent_history)
        messages.append(self._build_user_message(prompt, img_context=img_context))
        return messages

    def rebuild_rag(self) -> None:
        self.rag_service.rebuild_rag()

    def reset_conversation(self) -> None:
        self.convo = [{"role": "system", "content": self._system_prompt()}]
        self._summary_text = ""
        self.llm_service.conversation_context.forget()

    def chat_stream_text(self, prompt: str, img_context: str = None, use_rag: bool = False, document: Optional[dict] = None):
        base_prompt = prompt.strip()
        provider_context = self.llm_service.gather_context(base_prompt)

        source_filter = None
        if document and isinstance(document, dict):
            source_filter = document.get("name") or None

        rag_context = ""
        if use_rag or self.rag_service.should_use_knowledge_base(base_prompt, document=document):
            rag_context = self.rag_service.retrieve_and_process(base_prompt, source_filter=source_filter)
            log(f"RAG context injected: {len(rag_context)} chars", title="RAG", style="bold cyan")

        message_payload = self._build_message_payload(
            base_prompt, img_context=img_context, provider_context=provider_context, rag_context=rag_context
        )

        full_response = ""
        for chunk in self.llm_service.stream_response(message_payload):
            full_response += chunk
            yield chunk

        self.convo.append(self._build_user_message(base_prompt, img_context=img_context))
        self.convo.append({"role": "assistant", "content": full_response})
        self.llm_service.conversation_context.add_exchange(base_prompt, full_response)
        self._summarize_history()
