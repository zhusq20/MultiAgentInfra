#!/usr/bin/env python3
import asyncio
import httpx
import time
import multiprocessing
import uvicorn
from opentinker.environment.gomoku.multi_agent_game_server import MultiAgentGameServer

async def run_server(port):
    server = MultiAgentGameServer()
    config = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="error")
    server_uvicorn = uvicorn.Server(config)
    await server_uvicorn.serve()

def start_server_process(port):
    p = multiprocessing.Process(target=lambda: asyncio.run(run_server(port)))
    p.start()
    return p

async def test_hang_reproduction():
    port = 8795
    server_process = start_server_process(port)
    time.sleep(2)  # Wait for server to start
    
    url = f"http://127.0.0.1:{port}"
    session_id = "test_hang_session"
    
    async with httpx.AsyncClient(timeout=10) as client:
        # 1. Create session
        await client.post(f"{url}/session/create", json={"session_id": session_id})
        
        # 2. Join Black
        await client.post(f"{url}/session/{session_id}/join", json={"agent_id": "black", "role": "BLACK"})
        
        # 3. Join White
        await client.post(f"{url}/session/{session_id}/join", json={"agent_id": "white", "role": "WHITE"})
        
        # 4. White starts waiting
        print("White agent starts waiting for Black's move...")
        white_wait_task = asyncio.create_task(
            client.get(f"{url}/session/{session_id}/wait_for_opponent", params={"agent_id": "white", "timeout": 5})
        )
        
        # 5. Black makes an INVALID move (out of bounds)
        print("Black makes an INVALID move...")
        black_move_resp = await client.post(
            f"{url}/session/{session_id}/move", 
            json={"agent_id": "black", "move": "99,99"}
        )
        print(f"Black move result: {black_move_resp.json()}")
        
        # 6. Check if White immediately wakes up
        print("Checking if White agent wakes up...")
        try:
            start_wait = time.time()
            white_resp = await asyncio.wait_for(white_wait_task, timeout=2)
            elapsed = time.time() - start_wait
            print(f"White agent woke up in {elapsed:.2f}s!")
            print(f"White wait result: {white_resp.json()}")
            
            if white_resp.json()["done"]:
                print("SUCCESS: White agent correctly detected game end!")
            else:
                print("FAILURE: White agent did not detect game end!")
        except asyncio.TimeoutError:
            print("FAILURE: White agent is still hanging!")
        
    server_process.terminate()
    server_process.join()

if __name__ == "__main__":
    asyncio.run(test_hang_reproduction())
