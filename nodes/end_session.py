from __future__ import annotations

import asyncio

import anthropic

from gen.messages_pb2 import ConvRequest, ConvResponse
from gen.axiom_context import AxiomContext


def end_session(ax: AxiomContext, input: ConvRequest) -> ConvResponse:
    """Read the full conversation history, generate a concise summary with an
    LLM, write that summary as a cross-session semantic memory, then formally
    close the session to trigger consolidation.

    Exercises:
      ax.agent.memory.session(id).history.last(n) — read full episodic history
      ax.agent.memory.write()                      — write cross-session semantic memory
      ax.agent.memory.session(id).end()            — close session + trigger consolidation
    """
    api_key, ok = ax.secrets.get("ANTHROPIC_API_KEY")
    if not ok:
        raise RuntimeError("ANTHROPIC_API_KEY secret is not configured")

    async def _run() -> ConvResponse:
        session = ax.agent.memory.session(input.session_id)

        # Read the full history for this session (up to 200 turns).
        turns = await session.history.last(200)

        if not turns:
            ax.log.info("end_session: no turns found, closing empty session", session_id=input.session_id)
            await session.end()
            return ConvResponse(
                session_id=input.session_id,
                response="Session closed. No conversation history was found.",
            )

        transcript = "\n".join(f"{t.role}: {t.content}" for t in turns)

        client = anthropic.Anthropic(api_key=api_key)
        summary_reply = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarise the following conversation in 2-3 concise sentences, "
                        "capturing the key topics and any decisions or outcomes reached:\n\n"
                        + transcript
                    ),
                }
            ],
        )
        summary = summary_reply.content[0].text

        # Write the summary as a cross-session (flow-scoped) semantic memory so it
        # is discoverable in future sessions without the session_id prefix.
        await ax.agent.memory.write(summary, importance=0.8)

        # Formally close the session; the platform will consolidate episodic turns.
        await session.end()

        ax.log.info(
            "end_session completed",
            session_id=input.session_id,
            turns_summarised=len(turns),
        )

        return ConvResponse(
            session_id=input.session_id,
            response=f"Session closed. Summary: {summary}",
        )

    return asyncio.run(_run())
