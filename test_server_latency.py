
import asyncio
import httpx
import time

async def test_session_create():
    url = "http://localhost:8792/session/create"
    session_id = f"test_debug_{int(time.time())}"
    print(f"Testing session creation: {session_id}")
    
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        try:
            resp = await client.post(url, json={
                "session_id": session_id,
                "board_size": 9,
                "max_total_steps": 40
            }, timeout=10.0)
            elapsed = time.time() - start_time
            print(f"Response ({resp.status_code}) in {elapsed:.4f}s")
            print(f"Body: {resp.json()}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Failed after {elapsed:.4f}s: {e}")

if __name__ == "__main__":
    asyncio.run(test_session_create())
