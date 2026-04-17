import logging
import os
import uuid

import aiohttp
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    RunContext,
    TurnHandlingOptions,
)
from livekit.agents.llm import function_tool
from livekit.plugins import noise_cancellation, silero, deepgram, google
from livekit.agents import llm, tts, stt
from livekit.agents import AgentStateChangedEvent, MetricsCollectedEvent, metrics
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN


logger = logging.getLogger("agent")

load_dotenv(".env.local")


from system_prompt import SYSTEM_PROMPT, WELCOME_MESSAGE
from tools.agent_tools import AgentToolsMixin

class Assistant(Agent, AgentToolsMixin):
    def __init__(self, room) -> None:
        super().__init__(
            instructions=SYSTEM_PROMPT,
        )
        self.room = room


class GroqLLM(llm.LLM):
    def __init__(self, *, api_key: str, model: str) -> None:
        super().__init__()
        self._api_key = api_key
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "groq"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls=NOT_GIVEN,
        tool_choice=NOT_GIVEN,
        extra_kwargs=NOT_GIVEN,
    ) -> llm.LLMStream:
        return _GroqLLMStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)

    async def aclose(self) -> None:
        return None


class _GroqLLMStream(llm.LLMStream):
    def _build_payload_messages(self) -> list[dict[str, str]]:
        # Keep only recent turns so prompt-token count stays low for faster TTFT.
        max_messages = int(os.getenv("GROQ_MAX_CONTEXT_MESSAGES", "8"))
        max_chars = int(os.getenv("GROQ_MAX_CONTEXT_CHARS", "5000"))

        history: list[dict[str, str]] = []
        for msg in self._chat_ctx.messages():
            text = msg.text_content
            if not text:
                continue
            history.append({"role": msg.role, "content": text})

        if len(history) > max_messages:
            history = history[-max_messages:]

        total_chars = sum(len(m["content"]) for m in history)
        while len(history) > 1 and total_chars > max_chars:
            removed = history.pop(0)
            total_chars -= len(removed["content"])

        return history

    async def _run(self) -> None:
        payload_messages = self._build_payload_messages()

        payload = {
            "model": self._llm.model,
            "messages": payload_messages,
            "temperature": 0.4,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._llm._api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(f"Groq API error {resp.status}: {body}")
                data = await resp.json()

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        usage_data = data.get("usage") or {}
        usage = llm.CompletionUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id=data.get("id", f"groq-{uuid.uuid4().hex[:8]}"),
                delta=llm.ChoiceDelta(role="assistant", content=content),
                usage=usage,
            )
        )


class FallbackLLM(llm.LLM):
    def __init__(self, *, primary: llm.LLM, fallback: llm.LLM) -> None:
        super().__init__()
        self._primary = primary
        self._fallback = fallback

    @property
    def model(self) -> str:
        return self._primary.model

    @property
    def provider(self) -> str:
        return self._primary.provider

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls=NOT_GIVEN,
        tool_choice=NOT_GIVEN,
        extra_kwargs=NOT_GIVEN,
    ) -> llm.LLMStream:
        return _FallbackLLMStream(
            self,
            primary=self._primary,
            fallback=self._fallback,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
        )

    async def aclose(self) -> None:
        await self._primary.aclose()
        await self._fallback.aclose()


class _FallbackLLMStream(llm.LLMStream):
    def __init__(
        self,
        llm_parent: FallbackLLM,
        *,
        primary: llm.LLM,
        fallback: llm.LLM,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
        parallel_tool_calls=NOT_GIVEN,
        tool_choice=NOT_GIVEN,
        extra_kwargs=NOT_GIVEN,
    ) -> None:
        super().__init__(llm_parent, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._primary = primary
        self._fallback = fallback
        self._parallel_tool_calls = parallel_tool_calls
        self._tool_choice = tool_choice
        self._extra_kwargs = extra_kwargs

    async def _forward(self, provider: llm.LLM) -> None:
        stream = provider.chat(
            chat_ctx=self._chat_ctx,
            tools=self._tools,
            conn_options=self._conn_options,
            parallel_tool_calls=self._parallel_tool_calls,
            tool_choice=self._tool_choice,
            extra_kwargs=self._extra_kwargs,
        )
        async with stream:
            async for chunk in stream:
                self._event_ch.send_nowait(chunk)

    async def _run(self) -> None:
        try:
            await self._forward(self._primary)
        except Exception as primary_error:
            logger.warning("Primary LLM failed; falling back to Gemini: %s", primary_error)
            await self._forward(self._fallback)


async def entrypoint(ctx: JobContext):
    llm_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    stt_model = os.getenv("DEEPGRAM_STT_MODEL", "nova-2")
    tts_model = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")
    stt_base_url = os.getenv("DEEPGRAM_STT_BASE_URL", "https://api.deepgram.com/v1/listen")
    primary_llm: llm.LLM
    gemini_llm = google.LLM(model=llm_model)
    if groq_api_key:
        primary_llm = FallbackLLM(
            primary=GroqLLM(api_key=groq_api_key, model=groq_model),
            fallback=gemini_llm,
        )
    else:
        primary_llm = gemini_llm

    session = AgentSession(
        # Current livekit deepgram STT plugin uses v1 listen params; nova-3 is compatible.
        stt=deepgram.STT(model=stt_model, language="en", base_url=stt_base_url),
        llm=primary_llm,
        # English voice output via Deepgram Aura.
        tts=deepgram.TTS(model=tts_model),
        vad=silero.VAD.load(),
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            endpointing={
                "min_delay": float(os.getenv("EOU_MIN_DELAY_SEC", "0.15")),
                "max_delay": float(os.getenv("EOU_MAX_DELAY_SEC", "0.6")),
            },
        ),
        # Favor latency over token efficiency for a more conversational feel.
        preemptive_generation=True,
    )
    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)


    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("Usage summary: %s", summary)


    ctx.add_shutdown_callback(log_usage)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if (
            ev.new_state == "speaking"
            and last_eou_metrics
            and session.current_speech
            and getattr(last_eou_metrics, "speech_id", None) == session.current_speech.id
        ):
            # Try to grab the time it finished speaking
            speaking_time = getattr(last_eou_metrics, "timestamp", getattr(last_eou_metrics, "end_of_utterance_time", ev.created_at))
            delta = ev.created_at - speaking_time
            delta_ms = delta * 1000 if isinstance(delta, (int, float)) else delta.total_seconds() * 1000
            logger.info("Time to first audio frame: %sms", delta_ms)

    

    agent = Assistant(room=ctx.room)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    # Speak the welcome/first message immediately after connection
    session.say(WELCOME_MESSAGE, allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
