"""
title: SMPTE Copilot RAG
description: Pipe for SMPTE-Copilot with clickable citations support
author: SMPTE
version: 1.0.0
license: MIT
"""

from collections.abc import Awaitable, Callable

import aiohttp


class Pipe:
    """SMPTE Copilot Pipe for OpenWebUI with citation support.

    This Pipe connects to the SMPTE-Copilot RAG backend and emits
    citation events for each retrieved chunk, enabling clickable
    references in the OpenWebUI chat interface.
    """

    def __init__(self):
        self.type = "pipe"
        self.id = "smpte_copilot"
        self.name = "SMPTE Copilot"
        # Disable automatic citation to use our custom citation events
        self.citation = False

    class Valves:
        """Configuration options for the Pipe."""

        def __init__(self):
            self.SMPTE_API_BASE_URL = "http://api:8000"
            self.REQUEST_TIMEOUT = 120

    def pipes(self) -> list[dict]:
        """Return the list of available models/pipes."""
        return [{"id": "smpte_copilot_rag", "name": " - RAG"}]

    async def pipe(
        self,
        body: dict,
        __user__: dict | None = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] | None = None,
    ) -> str:
        """Process a chat request through the SMPTE RAG backend."""

        # OpenWebUI calls pipe twice: once with stream=True (main), once with stream=False (completed)
        # Only process on the first call (stream=True), skip the second to avoid duplicate work
        if not body.get("stream", True):
            # Second call (stream=False) - return last assistant message if exists
            messages = body.get("messages", [])
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
            return ""

        valves = self.Valves()

        messages = body.get("messages", [])
        if not messages:
            return "No messages provided."

        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            return "No user message found."

        query = user_messages[-1].get("content", "")
        if not query:
            return "Empty query."

        headers = {"Content-Type": "application/json"}

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": "Searching documents...", "done": False}
            })

        if __user__:
            if __user__.get("email"):
                headers["X-OpenWebUI-User-Email"] = __user__["email"]
            if __user__.get("id"):
                headers["X-OpenWebUI-User-Id"] = __user__["id"]
            if __user__.get("name"):
                headers["X-OpenWebUI-User-Name"] = __user__["name"]
            if __user__.get("role"):
                headers["X-OpenWebUI-User-Role"] = __user__["role"]

        try:
            async with aiohttp.ClientSession() as session, session.post(
                f"{valves.SMPTE_API_BASE_URL}/v1/rag/query",
                json={"query": query},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=valves.REQUEST_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return f"Error from SMPTE API: {resp.status} - {error_text}"

                data = await resp.json()

        except aiohttp.ClientError as e:
            return f"Failed to connect to SMPTE API: {e}"
        except Exception as e:
            return f"Unexpected error: {e}"

        response_text = data.get("response", "No response received.")
        citations = data.get("citations", [])

        if __event_emitter__ and citations:
            for citation in citations:
                source_name = citation.get("source") or "Unknown source"
                page = citation.get("page")
                content = citation.get("content", "")

                source_display = source_name
                if page is not None:
                    source_display = f"{source_name} (page {page})"

                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "document": [content],
                            "metadata": [
                                {
                                    "source": source_name,
                                    "page": page,
                                    "score": citation.get("score"),
                                }
                            ],
                            "source": {"name": source_display},
                        },
                    }
                )

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": "Completed!", "done": True}
            })

        return response_text
