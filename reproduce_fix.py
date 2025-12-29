import asyncio
import logging
import sys
from opentinker.environment.gomoku.multi_agent_gomoku_interaction import MultiAgentGomokuInteraction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def run_agent(role, session_id):
    interaction = MultiAgentGomokuInteraction(
        game_server_url="http://localhost:8790",
        agent_role=role,
        session_id=session_id,
        timeout=15 
    )
    
    print(f"[{role}] Initializing...")
    await interaction.start_interaction(request_id=f"req_{role}", game_session_id=session_id)
    
    # Simulate a few turns
    moves = {
        "BLACK": ["<move>4,4</move>", "<move>5,4</move>", "<move>6,4</move>"],
        "WHITE": ["<move>4,5</move>", "<move>5,5</move>", "<move>6,5</move>"]
    }
    
    try:
        if role == "BLACK":
            for i, move_text in enumerate(moves["BLACK"]):
                print(f"[{role}] Turn {i+1}: Making move {move_text}...")
                done, obs, reward, info = await interaction.generate_response(f"req_{role}", [{"role": "user", "content": move_text}])
                print(f"[{role}] Turn {i+1} result: done={done}, reward={reward}")
                if done: break
        else:
            # WHITE already waited for BLACK's first move in start_interaction
            for i, move_text in enumerate(moves["WHITE"]):
                print(f"[{role}] Turn {i+1}: Making response {move_text}...")
                # Note: For WHITE, the first generate_response starts AFTER BLACK's first move
                done, obs, reward, info = await interaction.generate_response(f"req_{role}", [{"role": "user", "content": move_text}])
                print(f"[{role}] Turn {i+1} result: done={done}, reward={reward}")
                if done: break
    except Exception as e:
        print(f"[{role}] Error: {e}")
    finally:
        await interaction.client.aclose()
        print(f"[{role}] Finished.")

async def main():
    session_id = "test_sync_v1"
    # Run both agents concurrently
    await asyncio.gather(
        run_agent("BLACK", session_id),
        run_agent("WHITE", session_id)
    )

if __name__ == "__main__":
    asyncio.run(main())
