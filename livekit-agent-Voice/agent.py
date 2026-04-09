import logging
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
)
from livekit.agents.llm import function_tool
from livekit.plugins import noise_cancellation, silero
from livekit.agents import llm,tts,stt,inference
from livekit.agents import AgentStateChangedEvent, MetricsCollectedEvent, metrics
import time


logger = logging.getLogger("agent")

load_dotenv(".env.local")


from system_prompt import SYSTEM_PROMPT
from tools.agent_tools import AgentToolsMixin

class Assistant(Agent, AgentToolsMixin):
    def __init__(self, room) -> None:
        super().__init__(
            instructions=SYSTEM_PROMPT,
        )
        self.room = room


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt="deepgram/nova-3",
        llm="google/gemini-2.5-flash",
        tts="cartesia/sonic-2:a167e0f3-df7e-4d52-a9c3-f949145efdab",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
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

    

    await session.start(
        agent=Assistant(room=ctx.room),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
