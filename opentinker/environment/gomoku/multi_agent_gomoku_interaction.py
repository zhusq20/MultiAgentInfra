#!/usr/bin/env python3
# Copyright 2025 OpenTinker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-Agent Gomoku Interaction for LLM vs LLM games.

This module provides an interaction class for multi-agent Gomoku where
two LLM agents play against each other via a shared game server.

Usage:
    # In interaction config YAML:
    interaction:
      - name: multi_agent_gomoku
        class_path: opentinker.environment.gomoku.multi_agent_gomoku_interaction.MultiAgentGomokuInteraction
        config:
          game_server_url: http://localhost:8790
          agent_role: BLACK  # or WHITE
"""

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from verl.interactions.base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class MultiAgentGomokuInteraction(BaseInteraction):
    """Interaction for multi-agent Gomoku where opponent is another LLM.
    
    This class connects to a MultiAgentGameServer and handles:
    - Joining game sessions
    - Submitting moves
    - Waiting for opponent moves
    - Turn-based synchronization
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        game_server_url: str = "http://localhost:8790",
        agent_role: str = "BLACK",
        session_id: str = "default_session",
        board_size: int = 9,
        max_total_steps: int = 40,
        timeout: float = 300,
        **kwargs,
    ):
        """Initialize the multi-agent Gomoku interaction.
        
        Args:
            config: Configuration dict (when loaded by verl's interaction_registry)
            game_server_url: URL of the MultiAgentGameServer
            agent_role: Role of this agent ("BLACK" or "WHITE")
            session_id: Session ID for game pairing
            board_size: Size of the game board
            max_total_steps: Maximum moves before game ends
            timeout: Timeout for waiting operations
        """
        # BaseInteraction requires config as positional argument
        super().__init__(config=config or {})
        
        # If config is provided (loaded by verl's interaction_registry), extract params
        if config:
            game_server_url = config.get("game_server_url", game_server_url)
            agent_role = config.get("agent_role", agent_role)
            session_id = config.get("session_id", session_id)
            board_size = config.get("board_size", board_size)
            max_total_steps = config.get("max_total_steps", max_total_steps)
            timeout = config.get("timeout", timeout)
        
        self.game_server_url = game_server_url.rstrip("/")
        self.agent_role = agent_role.upper()
        self.session_id = session_id
        self.board_size = board_size
        self.max_total_steps = max_total_steps
        self.timeout = timeout
        
        # Session tracking: request_id -> session_id
        self._sessions: Dict[str, str] = {}
        
        # Instance data: request_id -> instance state
        self._instance_dict: Dict[str, Dict[str, Any]] = {}
        
        self.client = httpx.AsyncClient(timeout=self.timeout)
        
        logger.info(
            f"MultiAgentGomokuInteraction initialized: role={self.agent_role}, "
            f"server={self.game_server_url}, session={self.session_id}"
        )
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to the game server."""
        url = f"{self.game_server_url}/{endpoint}"
        timeout = timeout or self.timeout

        try:
            if method.upper() == "GET":
                response = await self.client.get(url, params=params, timeout=timeout)
            elif method.upper() == "POST":
                response = await self.client.post(url, json=json_data, params=params, timeout=timeout)
            elif method.upper() == "DELETE":
                response = await self.client.delete(url, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Request to {endpoint} failed: {e}")
            raise
    
    async def start_interaction(
        self,
        request_id: str,
        **kwargs,
    ) -> None:
        """Start a new game interaction.
        
        Args:
            request_id: Unique identifier for this rollout
            **kwargs: May include:
                - game_session_id: Deterministic session ID for multi-agent pairing
                  (generated from step, sample_index, rollout_n)
                - session_id: Base session ID prefix (defaults to self.session_id)
                - agent_role: (optional) Override agent role
        """
        # Use deterministic game_session_id if provided (for multi-agent training)
        # This ensures BLACK and WHITE agents join the same session
        game_session_id = kwargs.get("game_session_id")
        
        if game_session_id:
            # Use the deterministic session ID directly
            session_id = game_session_id
            logger.info(f"[{request_id}] Using deterministic session_id: {session_id}")
        else:
            # Fallback: generate from request_id (for single-agent or testing)
            base_session_id = kwargs.get("session_id", self.session_id)
            if not base_session_id or base_session_id == "default_session":
                base_session_id = "game"
            session_id = f"{base_session_id}_{request_id}"
            logger.info(f"[{request_id}] Generated session_id from request_id: {session_id}")
        
        agent_role = kwargs.get("agent_role", self.agent_role)
        agent_id = f"agent_{agent_role}_{session_id}"
        
        # Extract initial moves if provided (for training from partially played boards)
        initial_moves = kwargs.get("initial_moves")
        if not initial_moves and "env_kwargs" in kwargs:
            initial_moves = kwargs["env_kwargs"].get("initial_moves")
        
        # Try to create session first (will fail or reset if exists, which is OK)
        try:
            await self._make_request("POST", "session/create", json_data={
                "session_id": session_id,
                "board_size": self.board_size,
                "max_total_steps": self.max_total_steps,
                "initial_moves": initial_moves,
            })
            logger.info(f"[{request_id}] Created/Reset session {session_id} with {len(initial_moves or [])} initial moves")
        except httpx.HTTPStatusError as e:
            logger.error(f"[{request_id}] Failed to create session {session_id}: {e}")
            raise
        
        # Join the session
        join_result = await self._make_request("POST", f"session/{session_id}/join", json_data={
            "agent_id": agent_id,
            "role": agent_role,
        })
        
        logger.info(
            f"[{request_id}] Joined session {session_id} as {join_result['role']}. "
            f"Status: {join_result['status']}"
        )
        
        # Wait for game to start (opponent to join)
        if join_result["status"] != "in_progress":
            logger.info(f"[{request_id}] Waiting for opponent to join...")
            start_result = await self._make_request(
                "POST",
                f"session/{session_id}/wait_for_start",
                params={"agent_id": agent_id, "timeout": self.timeout},
                timeout=self.timeout + 5,
            )
            logger.info(f"[{request_id}] Game started! Session: {session_id}, Role: {start_result['your_role']}")
        
        # Store session mapping
        self._sessions[request_id] = session_id
        self._instance_dict[request_id] = {
            "session_id": session_id,
            "agent_id": agent_id,
            "agent_role": agent_role,
            "initial_board_state": join_result["board"],
        }
        
        # If WHITE, wait for BLACK's first move
        if agent_role == "WHITE":
            logger.info(f"[{request_id}] WHITE waiting for BLACK's first move...")
            wait_result = await self._make_request(
                "GET",
                f"session/{session_id}/wait_for_opponent",
                params={"agent_id": agent_id, "timeout": self.timeout},
                timeout=self.timeout + 5,
            )
            self._instance_dict[request_id]["initial_board_state"] = wait_result["observation"]
            logger.info(f"[{request_id}] BLACK moved. WHITE's turn now.")
    
    async def generate_response(
        self,
        request_id: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """Process LLM's move and get opponent's response.
        
        Args:
            request_id: Unique identifier for this rollout
            messages: Conversation history with LLM's move in last message
            
        Returns:
            Tuple of (should_terminate, observation, reward, info)
        """
        if request_id not in self._sessions:
            raise ValueError(f"No session found for request {request_id}")
        
        session_id = self._sessions[request_id]
        instance = self._instance_dict[request_id]
        agent_id = instance["agent_id"]
        agent_role = instance["agent_role"]
        
        # Extract move from LLM's last message
        last_message = messages[-1]["content"] if messages else ""
        
        # Submit move to game server with wait_for_opponent=True
        # This blocks until opponent moves, combining move+wait into atomic operation
        move_result = await self._make_request("POST", f"session/{session_id}/move", json_data={
            "agent_id": agent_id,
            "move": last_message,
            "wait_for_opponent": True,  # Block until opponent moves
        })
        
        if not move_result["valid"]:
            # Invalid move - game ends with penalty
            logger.warning(
                f"[{request_id}] Invalid move: {move_result.get('error')}. "
                f"Move text: {last_message[:100]}"
            )
            return (
                True,  # should_terminate
                f"Invalid move: {move_result.get('error')}\n{move_result['observation']}",
                move_result["reward"],
                move_result["info"],
            )
        
        if move_result["done"]:
            # Game ended (win, loss, or draw)
            result = move_result["info"].get("result", "unknown")
            
            # Check if we won or lost based on result
            if "win" in result.lower():
                # Game ended with a winner
                if agent_role.lower() in result.lower():
                    # We won
                    reward = move_result["reward"]
                else:
                    # Opponent won (we lost)
                    reward = -1.0
            else:
                # Draw or other terminal state
                reward = move_result["reward"]
            
            # Use next_observation if available, otherwise use observation
            final_observation = move_result.get("next_observation") or move_result["observation"]
            logger.info(f"[{request_id}] Game ended. Result: {result}, Session: {session_id}")
            return (
                True,
                f"Game Over - {result.upper()}\n{final_observation}",
                reward,
                move_result["info"],
            )
        
        # Game continues - opponent has moved (already waited via wait_for_opponent=True)
        opponent_move = move_result.get("opponent_move", {})
        next_observation = move_result.get("next_observation", move_result["observation"])
        
        opponent_pos = opponent_move.get("position", ["?", "?"])
        observation = (
            f"Opponent moved to ({opponent_pos[0]}, {opponent_pos[1]}). Your turn.\n"
            f"{next_observation}"
        )
        
        return (
            False,  # should_terminate
            observation,
            0.0,  # intermediate reward
            {"last_opponent_move": opponent_move},
        )

    async def finalize_interaction(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
        logger.info(f"MultiAgentGomokuInteraction finalized for session {self.session_id}")
    
    def get_system_prompt(self) -> str:
        """Return the system prompt for multi-agent Gomoku."""
        symbol = "X" if self.agent_role == "BLACK" else "O"
        opponent_symbol = "O" if self.agent_role == "BLACK" else "X"
        
        return f"""You are playing Gomoku (Five in a Row) as {self.agent_role} ({symbol}).
Your opponent is playing as {opponent_symbol}.
You need to place 5 {symbol}s in a row (horizontally, vertically, or diagonally) to win.
Prevent your opponent from achieving 5 in a row.

Board coordinates are (row, col), both starting from 0.
The board is {self.board_size}x{self.board_size}.

Respond with your thinking process and then your move in the format:
<thinking>Your analysis of the current position...</thinking>
<move>row,col</move>

Example: <thinking>The center is open, I'll take it.</thinking><move>4,4</move>"""
    
    def get_initial_user_message(self) -> str:
        """Return the initial user message."""
        if self.agent_role == "BLACK":
            return f"You play first as BLACK (X). The board is empty.\nMake your first move."
        else:
            return "You are WHITE (O). Wait for BLACK's first move."
