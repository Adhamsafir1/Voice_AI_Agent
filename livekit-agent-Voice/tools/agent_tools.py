from livekit.agents.llm import function_tool
from livekit.agents import RunContext
import chromadb
import logging
import asyncio

logger = logging.getLogger("agent_tools")

class AgentToolsMixin:
    """
    A mixin class containing the tools available to the AI Agent.
    Requires `self.room` to be set if tools interact with the room (e.g. ending calls).
    """

    @function_tool
    async def search_knowledge_base(self, context: RunContext, query: str) -> str:
        """Searches the knowledge base document for answers. Call this immediately when the user asks a factual question."""
        logger.info("Accessing ChromaDB knowledge base for user query: %s", query)
        try:
            client = chromadb.PersistentClient(path="./.chroma_db")
            collection = client.get_collection(name="voice_agent_kb")
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            if not results["documents"] or not results["documents"][0]:
                return "No relevant information found in the knowledge base."
                
            # Combine the top results into a single context string
            context_string = "\n\n".join(results["documents"][0])
            return f"Context found:\n{context_string}"
        except Exception as e:
            logger.error("Error accessing vector database: %s", e)
            return "Knowledge base database is unavailable or missing."

    @function_tool
    async def end_call(self, context: RunContext) -> str:
        """Ends the call. Call this function exactly when the user says goodbye, bye, or indicates they want to hang up or leave."""
        logger.info("User requested to end the call. Disconnecting...")
        if hasattr(self, 'room') and self.room:
            # We schedule the room disconnect so the agent has a split second 
            # to finish its goodbye sentence before it abruptly cuts off.
            async def delayed_disconnect():
                await asyncio.sleep(3.0)
                await self.room.disconnect()
                import os
                import signal
                # Simulate pressing Ctrl+C so the Livekit agent shuts down cleanly 
                # without throwing nasty IPC socket crash errors!
                os.kill(os.getpid(), signal.SIGINT) 
                
            asyncio.create_task(delayed_disconnect())
            return "Ending the call now. Say your goodbye!"
        
        return "Failed to end call: No room context."
