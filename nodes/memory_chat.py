from __future__ import annotations

import asyncio

import anthropic

from gen.messages_pb2 import ConvRequest, ConvResponse
from gen.axiom_context import AxiomContext


def memory_chat(ax: AxiomContext, input: ConvRequest) -> ConvResponse:
    """Load session history and semantic context, call an LLM, then persist the
    new conversation turns and any notable facts back to memory.

    Exercises:
      ax.agent.memory.session(id).history.last(n)   — read episodic history
      ax.agent.memory.session(id).search(query)     — read semantic context
      ax.agent.memory.session(id).history.append()  — write episodic history
      ax.agent.memory.session(id).write()            — write semantic memory
    """
    api_key, ok = ax.secrets.get("ANTHROPIC_API_KEY")
    if not ok:
        raise RuntimeError("ANTHROPIC_API_KEY secret is not configured")

    async def _run() -> ConvResponse:
        session = ax.agent.memory.session(input.session_id)

        # Load the most recent conversation turns for this session.
        turns = await session.history.last(20)

        # Search for relevant semantic memories that might provide useful context.
        memories = await session.search(input.user_message, limit=5)

        # Build the messages array for the Claude API.
        # Prepend any retrieved semantic memories as a synthetic exchange so the
        # model receives relevant long-term context without polluting the turn list.
        messages = []
        if memories:
            context_lines = "\n".join(f"- {m.content}" for m in memories)
            messages.append({"role": "user", "content": f"[Relevant context from memory]\n{context_lines}"})
            messages.append({"role": "assistant", "content": "Understood, I will use this context."})

        for turn in turns:
            if turn.role in ("user", "assistant"):
                messages.append({"role": turn.role, "content": turn.content})

        messages.append({"role": "user", "content": input.user_message})

        client = anthropic.Anthropic(api_key=api_key)
        reply = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=messages,
        )
        response_text = reply.content[0].text

        # Persist both sides of the exchange to episodic history.
        await session.history.append(role="user", content=input.user_message)
        await session.history.append(role="assistant", content=response_text)

        # Write a semantic memory entry so future searches can surface this exchange.
        await session.write(
            f"User: {input.user_message[:300]} | Assistant: {response_text[:300]}",
            importance=0.6,
        )

        ax.log.info(
            "memory_chat completed",
            session_id=input.session_id,
            history_turns=len(turns),
            semantic_hits=len(memories),
        )

        return ConvResponse(session_id=input.session_id, response=response_text)

    return asyncio.run(_run())
