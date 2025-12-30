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
"""Multi-Agent Game Server for Gomoku.

This module provides an HTTP server that manages shared game state for
multi-agent Gomoku games. Two LLM agents can play against each other
with proper turn synchronization.

Usage:
    # Start the server
    python -m opentinker.environment.gomoku.multi_agent_game_server --port 8790
    
    # Or import and use programmatically
    from opentinker.environment.gomoku.multi_agent_game_server import MultiAgentGameServer
    server = MultiAgentGameServer()
    server.run(port=8790)
"""

import asyncio
import logging
import os
import re
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class PlayerRole(str, Enum):
    """Player roles in Gomoku."""
    BLACK = "BLACK"  # X, plays first
    WHITE = "WHITE"  # O, plays second


class GameStatus(str, Enum):
    """Game status."""
    WAITING = "waiting"  # Waiting for players to join
    IN_PROGRESS = "in_progress"  # Game is ongoing
    BLACK_WIN = "black_win"
    WHITE_WIN = "white_win"
    DRAW = "draw"


@dataclass
class GameSession:
    """Manages a single game between two agents."""

    session_id: str
    board_size: int = 9
    win_length: int = 5
    max_total_steps: int = 40
    max_invalid_moves: int = 3  # Max invalid moves before game ends

    # Game state
    board: List[List[str]] = field(default_factory=list)
    current_turn: PlayerRole = PlayerRole.BLACK
    move_history: List[Dict[str, Any]] = field(default_factory=list)
    status: GameStatus = GameStatus.WAITING
    step_count: int = 0

    # Agent tracking
    agents_joined: Dict[str, PlayerRole] = field(default_factory=dict)  # agent_id -> role
    invalid_move_counts: Dict[str, int] = field(default_factory=dict)  # agent_id -> count

    # Initial state
    initial_moves: List[List[int]] = field(default_factory=list)

    # Synchronization
    move_event: asyncio.Event = field(default_factory=asyncio.Event)
    join_event: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    def __post_init__(self):
        """Initialize the game board."""
        if not self.board:
            self.board = [["." for _ in range(self.board_size)] for _ in range(self.board_size)]
        
        # Apply initial moves if any
        if self.initial_moves:
            self._apply_initial_moves()
    
    def _apply_initial_moves(self):
        """Apply initial moves to the board and set current turn."""
        for i, move in enumerate(self.initial_moves):
            if len(move) >= 2:
                r, c = move[0], move[1]
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    symbol = "X" if i % 2 == 0 else "O"
                    self.board[r][c] = symbol
                    self.step_count += 1
                    self.move_history.append({
                        "step": self.step_count,
                        "agent": PlayerRole.BLACK.value if i % 2 == 0 else PlayerRole.WHITE.value,
                        "position": [r, c],
                        "symbol": symbol,
                        "is_initial": True
                    })
        
        # Set next turn
        self.current_turn = PlayerRole.BLACK if len(self.initial_moves) % 2 == 0 else PlayerRole.WHITE
    
    def reset(self, initial_moves: Optional[List[List[int]]] = None):
        """Reset the game to initial state."""
        if initial_moves is not None:
            self.initial_moves = initial_moves

        self.board = [["." for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.move_history = []
        self.step_count = 0
        self.invalid_move_counts = {}  # Reset invalid move counters

        if self.initial_moves:
            self._apply_initial_moves()
        else:
            self.current_turn = PlayerRole.BLACK

        self.status = GameStatus.IN_PROGRESS if len(self.agents_joined) == 2 else GameStatus.WAITING
        self.move_event = asyncio.Event()
    
    def render_board(self) -> str:
        """Render the board as a text string."""
        lines = []
        # Column headers
        header = "    " + " ".join(str(i) for i in range(self.board_size))
        lines.append(header)
        
        for i, row in enumerate(self.board):
            row_str = f"  {i} " + " ".join(row)
            lines.append(row_str)
        
        return "\n".join(lines)
    
    def get_state_for_agent(self, agent_role: PlayerRole) -> Dict[str, Any]:
        """Get game state from an agent's perspective."""
        return {
            "board": self.render_board(),
            "board_raw": [row[:] for row in self.board],
            "current_turn": self.current_turn.value,
            "your_role": agent_role.value,
            "your_symbol": "X" if agent_role == PlayerRole.BLACK else "O",
            "is_your_turn": self.current_turn == agent_role,
            "move_history": self.move_history,
            "step_count": self.step_count,
            "status": self.status.value,
        }
    
    def apply_move(self, row: int, col: int, agent_role: PlayerRole) -> Dict[str, Any]:
        """Apply a move to the board.

        Note: Validation is done in make_move before calling this method.

        Returns:
            Dict with observation, reward, done, and info
        """
        symbol = "X" if agent_role == PlayerRole.BLACK else "O"

        # Apply move (validation already done in make_move)
        self.board[row][col] = symbol
        self.step_count += 1
        self.move_history.append({
            "step": self.step_count,
            "agent": agent_role.value,
            "position": [row, col],
            "symbol": symbol,
        })
        
        # Check for win
        if self._check_win(row, col, symbol):
            self.status = GameStatus.BLACK_WIN if agent_role == PlayerRole.BLACK else GameStatus.WHITE_WIN
            return {
                "valid": True,
                "observation": self.render_board(),
                "reward": 1.0,
                "done": True,
                "info": {"result": "win", "winner": agent_role.value},
            }
        
        # Check for draw (board full or max steps reached)
        if self.step_count >= self.max_total_steps or self._is_board_full():
            self.status = GameStatus.DRAW
            return {
                "valid": True,
                "observation": self.render_board(),
                "reward": -0.1,  # Small penalty for draw - encourage winning
                "done": True,
                "info": {"result": "draw"},
            }
        
        # Game continues - switch turn
        self.current_turn = PlayerRole.WHITE if agent_role == PlayerRole.BLACK else PlayerRole.BLACK
        
        return {
            "valid": True,
            "observation": self.render_board(),
            "reward": 0.0,
            "done": False,
            "info": {"result": "continue"},
        }
    
    def _check_win(self, row: int, col: int, symbol: str) -> bool:
        """Check if the last move resulted in a win."""
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal
            (1, -1),  # anti-diagonal
        ]
        
        for dr, dc in directions:
            count = 1
            # Count in positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == symbol:
                count += 1
                r, c = r + dr, c + dc
            
            # Count in negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == symbol:
                count += 1
                r, c = r - dr, c - dc
            
            if count >= self.win_length:
                return True
        
        return False
    
    def _is_board_full(self) -> bool:
        """Check if the board is full."""
        for row in self.board:
            if "." in row:
                return False
        return True


# ==================== Pydantic Models ====================

class CreateSessionRequest(BaseModel):
    """Request to create a new game session."""
    session_id: str
    board_size: int = 9
    win_length: int = 5
    max_total_steps: int = 40
    initial_moves: Optional[List[List[int]]] = None


class CreateSessionResponse(BaseModel):
    """Response after creating a session."""
    session_id: str
    status: str
    message: str


class JoinSessionRequest(BaseModel):
    """Request to join a game session."""
    agent_id: str
    role: Optional[str] = None  # If not specified, auto-assign


class JoinSessionResponse(BaseModel):
    """Response after joining a session."""
    session_id: str
    agent_id: str
    role: str
    status: str
    board: str
    message: str


class MakeMoveRequest(BaseModel):
    """Request to make a move."""
    agent_id: str
    move: str  # Format: "row,col" or "<move>row,col</move>"
    wait_for_opponent: bool = False  # If True, block until opponent moves
    wait_timeout: float = 600  # Timeout for waiting for opponent (seconds)


class MakeMoveResponse(BaseModel):
    """Response after making a move."""
    valid: bool
    observation: str
    reward: float
    done: bool
    info: Dict[str, Any]
    error: Optional[str] = None
    # Fields for wait_for_opponent=True mode
    next_observation: Optional[str] = None  # Board state after opponent moved
    opponent_move: Optional[Dict[str, Any]] = None  # Opponent's move details


class WaitForOpponentResponse(BaseModel):
    """Response after waiting for opponent's move."""
    opponent_moved: bool
    observation: str
    last_move: Optional[Dict[str, Any]]
    done: bool
    info: Dict[str, Any]


class GameStateResponse(BaseModel):
    """Response with current game state."""
    session_id: str
    board: str
    current_turn: str
    status: str
    step_count: int
    move_history: List[Dict[str, Any]]


# ==================== Multi-Agent Game Server ====================

class MultiAgentGameServer:
    """HTTP server managing shared game state for multi-agent Gomoku."""
    
    def __init__(self):
        self.app = FastAPI(title="Multi-Agent Gomoku Game Server")
        self.sessions: Dict[str, GameSession] = {}
        self.lock = threading.Lock()
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "sessions": len(self.sessions)}
        
        @self.app.post("/session/create", response_model=CreateSessionResponse)
        async def create_session(request: CreateSessionRequest):
            """Create a new game session."""
            if request.session_id in self.sessions:
                session = self.sessions[request.session_id]
                async with session.lock:
                    # CRITICAL FIX: Only reset if no agents have joined or game is finished
                    # If game is in progress OR agents are waiting to start, do not reset.
                    # This prevents race conditions where:
                    # 1. BLACK joins, status=WAITING
                    # 2. WHITE calls create_session, would reset and clear BLACK's join
                    # 3. BLACK's wait_for_start times out or sees corrupted state
                    if session.status == GameStatus.IN_PROGRESS:
                        logger.info(f"Session {request.session_id} already in progress, skipping reset")
                        return CreateSessionResponse(
                            session_id=request.session_id,
                            status="in_progress",
                            message=f"Session {request.session_id} already in progress, not reset.",
                        )

                    if session.agents_joined:
                        # Some agents have joined but game hasn't started yet
                        # Don't reset - let the new agent join
                        logger.info(
                            f"Session {request.session_id} has {len(session.agents_joined)} agents waiting, "
                            f"skipping reset"
                        )
                        return CreateSessionResponse(
                            session_id=request.session_id,
                            status="waiting",
                            message=f"Session {request.session_id} has agents waiting, not reset.",
                        )

                    # Safe to reset if no agents have joined and game is not in progress
                    session.reset(initial_moves=request.initial_moves)

                logger.info(f"Reset existing session {request.session_id}")
                return CreateSessionResponse(
                    session_id=request.session_id,
                    status="reset",
                    message=f"Session {request.session_id} already exists, has been reset.",
                )
            
            session = GameSession(
                session_id=request.session_id,
                board_size=request.board_size,
                win_length=request.win_length,
                max_total_steps=request.max_total_steps,
                initial_moves=request.initial_moves or [],
            )
            self.sessions[request.session_id] = session
            
            logger.info(f"Created session {request.session_id}")
            return CreateSessionResponse(
                session_id=request.session_id,
                status="created",
                message=f"Session created. Waiting for players to join.",
            )
        
        @self.app.post("/session/{session_id}/join", response_model=JoinSessionResponse)
        async def join_session(session_id: str, request: JoinSessionRequest):
            """Join an existing game session."""
            if session_id not in self.sessions:
                raise HTTPException(404, f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            
            async with session.lock:
                # Check if agent already joined
                if request.agent_id in session.agents_joined:
                    role = session.agents_joined[request.agent_id]
                    return JoinSessionResponse(
                        session_id=session_id,
                        agent_id=request.agent_id,
                        role=role.value,
                        status=session.status.value,
                        board=session.render_board(),
                        message=f"Already joined as {role.value}",
                    )
                
                # Check if session is full
                if len(session.agents_joined) >= 2:
                    raise HTTPException(400, "Session is full")
                
                # Assign role
                if request.role:
                    role = PlayerRole(request.role)
                    if role in session.agents_joined.values():
                        raise HTTPException(400, f"Role {role.value} is already taken")
                else:
                    # Auto-assign: BLACK first, then WHITE
                    if PlayerRole.BLACK not in session.agents_joined.values():
                        role = PlayerRole.BLACK
                    else:
                        role = PlayerRole.WHITE
                
                session.agents_joined[request.agent_id] = role
                
                # If both players joined, start the game
                if len(session.agents_joined) == 2:
                    session.status = GameStatus.IN_PROGRESS
                    session.join_event.set()
                
                logger.info(f"Agent {request.agent_id} joined session {session_id} as {role.value}")
                return JoinSessionResponse(
                    session_id=session_id,
                    agent_id=request.agent_id,
                    role=role.value,
                    status=session.status.value,
                    board=session.render_board(),
                    message=f"Joined as {role.value}. {'Game started!' if session.status == GameStatus.IN_PROGRESS else 'Waiting for opponent...'}",
                )
        
        @self.app.post("/session/{session_id}/wait_for_start")
        async def wait_for_start(session_id: str, agent_id: str, timeout: float = 300):
            """Wait for all players to join and game to start."""
            if session_id not in self.sessions:
                raise HTTPException(404, f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            
            if session.status == GameStatus.IN_PROGRESS:
                role = session.agents_joined.get(agent_id)
                return {
                    "started": True,
                    "status": session.status.value,
                    "your_role": role.value if role else None,
                    "board": session.render_board(),
                }
            
            try:
                await asyncio.wait_for(session.join_event.wait(), timeout=timeout)
                role = session.agents_joined.get(agent_id)
                return {
                    "started": True,
                    "status": session.status.value,
                    "your_role": role.value if role else None,
                    "board": session.render_board(),
                }
            except asyncio.TimeoutError:
                raise HTTPException(408, f"Timeout waiting for game to start")
        
        @self.app.post("/session/{session_id}/move", response_model=MakeMoveResponse)
        async def make_move(session_id: str, request: MakeMoveRequest):
            """Make a move in the game."""
            if session_id not in self.sessions:
                raise HTTPException(404, f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            
            # Validate agent
            if request.agent_id not in session.agents_joined:
                raise HTTPException(400, f"Agent {request.agent_id} not in session")
            
            agent_role = session.agents_joined[request.agent_id]
            
            async with session.lock:
                # Validate game state
                if session.status != GameStatus.IN_PROGRESS:
                    return MakeMoveResponse(
                        valid=False,
                        observation=session.render_board(),
                        reward=0.0,
                        done=True,
                        info={"result": session.status.value},
                        error=f"Game is not in progress. Status: {session.status.value}",
                    )
                
                # Validate turn
                if session.current_turn != agent_role:
                    return MakeMoveResponse(
                        valid=False,
                        observation=session.render_board(),
                        reward=0.0,  # No penalty - simplified reward design
                        done=False,
                        info={"result": "not_your_turn"},
                        error=f"Not {agent_role.value}'s turn. Current turn: {session.current_turn.value}",
                    )
                
                # Parse move
                row, col = self._parse_move(request.move)
                if row is None or col is None:
                    # Track invalid move count
                    session.invalid_move_counts[request.agent_id] = \
                        session.invalid_move_counts.get(request.agent_id, 0) + 1
                    invalid_count = session.invalid_move_counts[request.agent_id]

                    if invalid_count >= session.max_invalid_moves:
                        # Too many invalid moves, end game
                        session.status = GameStatus.DRAW
                        # IMPORTANT: Notify opponent that game ended
                        session.move_event.set()
                        return MakeMoveResponse(
                            valid=False,
                            observation=session.render_board(),
                            reward=-0.5,  # Large penalty - 3 invalid moves = -2.0 > loss (-1.0)
                            done=True,
                            info={"result": "too_many_invalid_moves", "invalid_count": invalid_count},
                            error=f"Too many invalid moves ({invalid_count}). Game over.",
                        )
                    else:
                        # Allow retry
                        return MakeMoveResponse(
                            valid=False,
                            observation=session.render_board(),
                            reward=-0.5,  # Large penalty per invalid move - discourage invalid moves
                            done=False,  # Game continues, allow retry
                            info={"result": "parse_error", "invalid_count": invalid_count,
                                  "retries_left": session.max_invalid_moves - invalid_count},
                            error=f"Could not parse move: {request.move}. Please use format <move>row,col</move>. "
                                  f"Retries left: {session.max_invalid_moves - invalid_count}",
                        )

                # Check if position is valid before applying
                if row < 0 or row >= session.board_size or col < 0 or col >= session.board_size:
                    session.invalid_move_counts[request.agent_id] = \
                        session.invalid_move_counts.get(request.agent_id, 0) + 1
                    invalid_count = session.invalid_move_counts[request.agent_id]

                    if invalid_count >= session.max_invalid_moves:
                        session.status = GameStatus.DRAW
                        # IMPORTANT: Notify opponent that game ended
                        session.move_event.set()
                        return MakeMoveResponse(
                            valid=False,
                            observation=session.render_board(),
                            reward=-0.5,  # Large penalty - 3 invalid moves = -2.0 > loss (-1.0)
                            done=True,
                            info={"result": "too_many_invalid_moves", "invalid_count": invalid_count},
                            error=f"Too many invalid moves ({invalid_count}). Game over.",
                        )
                    else:
                        return MakeMoveResponse(
                            valid=False,
                            observation=session.render_board(),
                            reward=-0.5,  # Large penalty per invalid move - discourage invalid moves
                            done=False,
                            info={"result": "out_of_bounds", "invalid_count": invalid_count,
                                  "retries_left": session.max_invalid_moves - invalid_count},
                            error=f"Move ({row}, {col}) is out of bounds. Board is {session.board_size}x{session.board_size}. "
                                  f"Retries left: {session.max_invalid_moves - invalid_count}",
                        )

                if session.board[row][col] != ".":
                    session.invalid_move_counts[request.agent_id] = \
                        session.invalid_move_counts.get(request.agent_id, 0) + 1
                    invalid_count = session.invalid_move_counts[request.agent_id]

                    if invalid_count >= session.max_invalid_moves:
                        session.status = GameStatus.DRAW
                        # IMPORTANT: Notify opponent that game ended
                        session.move_event.set()
                        return MakeMoveResponse(
                            valid=False,
                            observation=session.render_board(),
                            reward=-0.5,  # Large penalty - 3 invalid moves = -2.0 > loss (-1.0)
                            done=True,
                            info={"result": "too_many_invalid_moves", "invalid_count": invalid_count},
                            error=f"Too many invalid moves ({invalid_count}). Game over.",
                        )
                    else:
                        return MakeMoveResponse(
                            valid=False,
                            observation=session.render_board(),
                            reward=-0.5,  # Large penalty per invalid move - discourage invalid moves
                            done=False,
                            info={"result": "occupied", "invalid_count": invalid_count,
                                  "retries_left": session.max_invalid_moves - invalid_count},
                            error=f"Position ({row}, {col}) is already occupied. "
                                  f"Retries left: {session.max_invalid_moves - invalid_count}",
                        )

                # Valid move - reset invalid move counter and apply
                session.invalid_move_counts[request.agent_id] = 0
                result = session.apply_move(row, col, agent_role)
                
                # Record current move count BEFORE setting event, used for wait loop below
                my_move_count = len(session.move_history)
                
                # Notify opponent that we moved
                session.move_event.set()
                
                logger.info(f"[{session_id}] {agent_role.value} moved to ({row}, {col}), move_count={my_move_count}")
                
                # If game ended or wait_for_opponent=False, return immediately
                if result["done"] or not request.wait_for_opponent:
                    return MakeMoveResponse(
                        valid=result["valid"],
                        observation=result["observation"],
                        reward=result["reward"],
                        done=result["done"],
                        info=result["info"],
                        error=result.get("error"),
                    )
            
            # wait_for_opponent=True: Wait for opponent's move outside the lock
            # This allows the opponent to acquire the lock and make their move
            timeout = request.wait_timeout  # Use configurable timeout
            start_time = time.time()
            logger.info(f"[{session_id}] {agent_role.value} waiting for opponent (timeout={timeout}s)")

            # RELIABLE WAIT STRATEGY:
            # Instead of relying solely on move_event (which can have race conditions),
            # we use move_history length as the authoritative signal.
            # When opponent moves, move_history length will increase beyond my_move_count.
            while True:
                # Check termination conditions
                if session.status != GameStatus.IN_PROGRESS:
                    break
                
                # Check if opponent has moved by comparing move_history length
                if len(session.move_history) > my_move_count:
                    # Opponent moved! Verify it's not somehow our own move (shouldn't happen)
                    last_move = session.move_history[-1]
                    if last_move.get("agent") != agent_role.value:
                        break  # Opponent moved, we can return
                    else:
                        # Safety: update my_move_count if somehow we moved again
                        my_move_count = len(session.move_history)
                
                # Check timeout
                remaining = timeout - (time.time() - start_time)
                if remaining <= 0:
                    raise HTTPException(408, "Timeout waiting for opponent's move")
                
                # Wait for move event signal (with short timeout to allow re-checking)
                try:
                    await asyncio.wait_for(session.move_event.wait(), timeout=min(remaining, 0.5))
                    session.move_event.clear()
                except asyncio.TimeoutError:
                    # Timeout is normal in polling loop, just re-check conditions
                    continue
            
            # Return the updated state after opponent moved
            opponent_last_move = None
            if session.move_history and len(session.move_history) > my_move_count:
                opponent_last_move = session.move_history[-1]
                # Double-check it's opponent's move
                if opponent_last_move.get("agent") == agent_role.value:
                    logger.warning(
                        f"[{session_id}] Last move is our own ({agent_role.value}), "
                        f"opponent may not have moved yet. move_history length: {len(session.move_history)}"
                    )
                    opponent_last_move = None

            return MakeMoveResponse(
                valid=result["valid"],
                observation=result["observation"],
                reward=result["reward"],
                done=session.status != GameStatus.IN_PROGRESS,
                info=result["info"] if session.status == GameStatus.IN_PROGRESS else {"result": session.status.value},
                error=result.get("error"),
                next_observation=session.render_board(),
                opponent_move=opponent_last_move,
            )
        
        @self.app.get("/session/{session_id}/wait_for_opponent", response_model=WaitForOpponentResponse)
        async def wait_for_opponent(session_id: str, agent_id: str, timeout: float = 300):
            """Wait for opponent to make a move."""
            if session_id not in self.sessions:
                raise HTTPException(404, f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            
            if agent_id not in session.agents_joined:
                raise HTTPException(400, f"Agent {agent_id} not in session")
            
            agent_role = session.agents_joined[agent_id]
            
            # If game ended, return immediately
            if session.status != GameStatus.IN_PROGRESS:
                return WaitForOpponentResponse(
                    opponent_moved=True,
                    observation=session.render_board(),
                    last_move=session.move_history[-1] if session.move_history else None,
                    done=True,
                    info={"result": session.status.value},
                )
            
            # If it's already our turn, opponent has moved
            if session.current_turn == agent_role:
                return WaitForOpponentResponse(
                    opponent_moved=True,
                    observation=session.render_board(),
                    last_move=session.move_history[-1] if session.move_history else None,
                    done=session.status != GameStatus.IN_PROGRESS,
                    info={"result": "continue" if session.status == GameStatus.IN_PROGRESS else session.status.value},
                )
            
            # Wait for opponent's move
            start_time = time.time()
            while session.current_turn != agent_role and session.status == GameStatus.IN_PROGRESS:
                try:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        raise asyncio.TimeoutError()

                    # Wait for move event with short timeout for robustness
                    await asyncio.wait_for(session.move_event.wait(), timeout=min(remaining, 0.5))
                    # Clear event after waking up to prepare for next iteration
                    session.move_event.clear()
                except asyncio.TimeoutError:
                    if time.time() - start_time >= timeout:
                        raise HTTPException(408, "Timeout waiting for opponent")
                    continue
            
            return WaitForOpponentResponse(
                opponent_moved=True,
                observation=session.render_board(),
                last_move=session.move_history[-1] if session.move_history else None,
                done=session.status != GameStatus.IN_PROGRESS,
                info={"result": "continue" if session.status == GameStatus.IN_PROGRESS else session.status.value},
            )
        
        @self.app.get("/session/{session_id}/state", response_model=GameStateResponse)
        async def get_state(session_id: str):
            """Get current game state."""
            if session_id not in self.sessions:
                raise HTTPException(404, f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            return GameStateResponse(
                session_id=session_id,
                board=session.render_board(),
                current_turn=session.current_turn.value,
                status=session.status.value,
                step_count=session.step_count,
                move_history=session.move_history,
            )
        
        @self.app.post("/session/{session_id}/reset")
        async def reset_session(session_id: str):
            """Reset a game session for a new game."""
            if session_id not in self.sessions:
                raise HTTPException(404, f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            async with session.lock:
                session.reset()
            
            logger.info(f"Reset session {session_id}")
            return {"status": "reset", "session_id": session_id}
        
        @self.app.post("/session/{session_id}/abort")
        async def abort_game(session_id: str, agent_id: str, reason: str = "client_abort"):
            """Abort an in-progress game.

            This is called when one agent needs to terminate early (e.g., sync error).
            It marks the game as DRAW and notifies the opponent so they can exit.
            """
            if session_id not in self.sessions:
                raise HTTPException(404, f"Session {session_id} not found")

            session = self.sessions[session_id]

            async with session.lock:
                if session.status != GameStatus.IN_PROGRESS:
                    # Game already ended, nothing to do
                    logger.info(
                        f"[{session_id}] Abort request ignored - game already ended: {session.status.value}"
                    )
                    return {
                        "status": session.status.value,
                        "message": f"Game already ended with status: {session.status.value}",
                    }

                # Get the aborting agent's role for logging
                aborting_role = session.agents_joined.get(agent_id, "unknown")
                opponent_role = "WHITE" if aborting_role == PlayerRole.BLACK else "BLACK"
                
                # Mark game as DRAW (abort is neither win nor loss)
                session.status = GameStatus.DRAW

                # CRITICAL: Notify opponent that game ended
                session.move_event.set()
                session.join_event.set()  # Also set join_event in case opponent is waiting to start

                logger.warning(
                    f"[{session_id}] GAME ABORTED by {aborting_role}. "
                    f"Reason: {reason}, move_count={session.step_count}, "
                    f"agents={list(session.agents_joined.keys())}. "
                    f"Opponent ({opponent_role}) notified."
                )

                return {
                    "status": "aborted",
                    "session_id": session_id,
                    "reason": reason,
                    "move_count": session.step_count,
                    "aborting_agent": agent_id,
                }

        @self.app.delete("/session/{session_id}")
        async def delete_session(session_id: str):
            """Delete a game session."""
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Deleted session {session_id}")
            return {"status": "deleted", "session_id": session_id}
    
    def _parse_move(self, move_str: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse move string to (row, col) tuple.
        
        Supports formats:
        - "row,col"
        - "<move>row,col</move>"
        - "<thinking>...</thinking><move>row,col</move>"
        """
        # Try to extract from <move> tags
        move_match = re.search(r'<move>\s*(\d+)\s*,\s*(\d+)\s*</move>', move_str)
        if move_match:
            return int(move_match.group(1)), int(move_match.group(2))
        
        # Try direct format
        direct_match = re.search(r'(\d+)\s*,\s*(\d+)', move_str)
        if direct_match:
            return int(direct_match.group(1)), int(direct_match.group(2))
        
        return None, None
    
    def run(self, host: str = "0.0.0.0", port: int = 8790):
        """Run the server."""
        logger.info(f"Starting Multi-Agent Game Server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Gomoku Game Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8790, help="Port to bind to")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    server = MultiAgentGameServer()
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
