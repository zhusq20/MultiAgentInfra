import asyncio
import httpx
import time
import subprocess
import os

async def run_agent(client, url, session_id, agent_id, role, num_moves):
    print(f"[{role}] Starting agent")
    for i in range(num_moves):
        # Determine if it's our turn
        # BLACK moves on 0, 2, 4... WHITE moves on 1, 3, 5...
        # Wait for our turn
        while True:
            resp = await client.get(f"{url}/session/{session_id}/state")
            state = resp.json()
            if state["status"] != "in_progress":
                print(f"[{role}] Game ended with status: {state['status']}")
                return
            if state["current_turn"] == role:
                break
            await asyncio.sleep(0.1)
        
        # Make move
        # Each agent makes up to 20 moves
        # BLACK moves: (0,0), (1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (0,2)...
        # WHITE moves: (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,0), (0,3)...
        if role == "BLACK":
            row, col = i % 9, (i * 2) % 9
        else:
            row, col = i % 9, (i * 2 + 1) % 9
        
        move_str = f"{row},{col}"
        
        print(f"[{role}] Making move {i}: {move_str}")
        # Use wait_for_opponent=True to test the fix
        try:
            resp = await client.post(f"{url}/session/{session_id}/move", 
                                   json={"agent_id": agent_id, "move": move_str, "wait_for_opponent": True},
                                   timeout=30)
            data = resp.json()
            print(f"[{role}] Move {i} response: done={data['done']}, reward={data['reward']}")
            
            if data["done"]:
                # If we made the 20th move (step 40), or opponent did, we should see -0.1 for draw
                if data["info"].get("result") == "draw":
                    if data["reward"] == -0.1:
                        print(f"[{role}] ✓ SUCCESS: Received draw reward -0.1")
                    else:
                        print(f"[{role}] ✗ FAILURE: Received reward {data['reward']} for draw")
                return
        except Exception as e:
            print(f"[{role}] Error during move: {e}")
            return

async def reproduce():
    server_port = 8796
    url = f"http://localhost:{server_port}"
    process = subprocess.Popen(
        ["python3", "opentinker/environment/gomoku/multi_agent_game_server.py", "--port", str(server_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    await asyncio.sleep(2)
    
    try:
        session_id = "repro_session_concurrent"
        async with httpx.AsyncClient() as client:
            await client.post(f"{url}/session/create", json={"session_id": session_id})
            await client.post(f"{url}/session/{session_id}/join", json={"agent_id": "black_1", "role": "BLACK"})
            await client.post(f"{url}/session/{session_id}/join", json={"agent_id": "white_1", "role": "WHITE"})
            
            print("\n--- Starting Concurrent Agents ---")
            await asyncio.gather(
                run_agent(client, url, session_id, "black_1", "BLACK", 20),
                run_agent(client, url, session_id, "white_1", "WHITE", 20)
            )
            
    finally:
        process.terminate()
        print("\nServer stopped.")

if __name__ == "__main__":
    asyncio.run(reproduce())
