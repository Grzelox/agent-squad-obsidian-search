from typing import Any, Dict, Optional


class QueryHandler:
    def __init__(self):
        self._next_handler: Optional[QueryHandler] = None

    def set_next(self, handler: "QueryHandler") -> "QueryHandler":
        self._next_handler = handler
        return handler

    def handle(self, question: str) -> Optional[Dict[str, Any]]:
        if self._next_handler:
            return self._next_handler.handle(question)
        return None


class ReActHandler(QueryHandler):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def handle(self, question: str) -> Optional[Dict[str, Any]]:
        should_use_tools = self.agent._should_use_function_calls(question)
        if should_use_tools and self.agent.agent_executor:
            self.agent.logger.debug("ReActHandler: trying ReAct agent")
            try:
                agent_result = self.agent.agent_executor.invoke({"input": question})
                if (
                    "intermediate_steps" in agent_result
                    and agent_result["intermediate_steps"]
                ):
                    output_lower = agent_result.get("output", "").strip().lower()
                    if (
                        output_lower.startswith("agent stopped")
                        or "iteration limit" in output_lower
                        or "time limit" in output_lower
                    ):
                        self.agent.logger.warning(
                            "Agent stopped due to iteration/time limit, passing to next handler."
                        )
                        return (
                            self._next_handler.handle(question)
                            if self._next_handler
                            else None
                        )
                    return {
                        "answer": agent_result["output"],
                        "sources": ["Function calls"],
                        "source_details": {
                            "original_sources": [],
                            "summary_sources": [],
                            "total_chunks": 0,
                            "used_function_calls": True,
                            "tools_used": [
                                step[0].tool
                                for step in agent_result["intermediate_steps"]
                            ],
                        },
                        "mode": "react",
                    }
            except Exception as e:
                self.agent.logger.error(f"ReActHandler failed: {str(e)}")
                return (
                    self._next_handler.handle(question) if self._next_handler else None
                )
        return super().handle(question)


class ManualFunctionCallHandler(QueryHandler):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def handle(self, question: str) -> Optional[Dict[str, Any]]:
        should_use_tools = self.agent._should_use_function_calls(question)
        if should_use_tools:
            self.agent.logger.debug(
                "ManualFunctionCallHandler: trying manual function call"
            )
            result = self.agent._handle_manual_function_call(question)
            if result["answer"]:
                return result
        return super().handle(question)


class RAGHandler(QueryHandler):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def handle(self, question: str) -> Optional[Dict[str, Any]]:
        self.agent.logger.debug("RAGHandler: using semantic search fallback")
        return self.agent._rag_answer(question)
