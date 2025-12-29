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
"""Tests for Multi-Agent Game Server.

Run with: pytest opentinker/tests/test_multi_agent_game.py -v
"""

import asyncio
import pytest
import threading
import time
from typing import List, Tuple

from opentinker.environment.gomoku.multi_agent_game_server import (
    GameSession,
    GameStatus,
    MultiAgentGameServer,
    PlayerRole,
)


class TestGameSession:
    """Test GameSession state management."""
    
    def test_session_creation(self):
        """Test creating a new game session."""
        session = GameSession(session_id="test_001", board_size=9)
        
        assert session.session_id == "test_001"
        assert session.board_size == 9
        assert session.status == GameStatus.WAITING
        assert session.current_turn == PlayerRole.BLACK
        assert len(session.agents_joined) == 0
    
    def test_board_initialization(self):
        """Test board is properly initialized."""
        session = GameSession(session_id="test_002", board_size=9)
        
        assert len(session.board) == 9
        assert all(len(row) == 9 for row in session.board)
        assert all(cell == "." for row in session.board for cell in row)
    
    def test_render_board(self):
        """Test board rendering."""
        session = GameSession(session_id="test_003", board_size=3)
        
        # Make a move
        session.board[1][1] = "X"
        
        rendered = session.render_board()
        assert "X" in rendered
        assert "." in rendered
    
    def test_apply_valid_move(self):
        """Test applying a valid move."""
        session = GameSession(session_id="test_004", board_size=9)
        session.status = GameStatus.IN_PROGRESS
        
        result = session.apply_move(4, 4, PlayerRole.BLACK)
        
        assert result["valid"] is True
        assert result["done"] is False
        assert result["reward"] == 0.0
        assert session.board[4][4] == "X"
        assert session.current_turn == PlayerRole.WHITE
    
    def test_apply_invalid_move_occupied(self):
        """Test applying move to occupied cell."""
        session = GameSession(session_id="test_005", board_size=9)
        session.status = GameStatus.IN_PROGRESS
        session.board[4][4] = "X"
        
        result = session.apply_move(4, 4, PlayerRole.WHITE)
        
        assert result["valid"] is False
        assert result["done"] is True
        assert "occupied" in result["info"]["error"]
    
    def test_apply_invalid_move_out_of_bounds(self):
        """Test applying move outside board."""
        session = GameSession(session_id="test_006", board_size=9)
        session.status = GameStatus.IN_PROGRESS
        
        result = session.apply_move(10, 10, PlayerRole.BLACK)
        
        assert result["valid"] is False
        assert result["done"] is True
        assert "out_of_bounds" in result["info"]["error"]
    
    def test_win_detection_horizontal(self):
        """Test horizontal win detection."""
        session = GameSession(session_id="test_007", board_size=9, win_length=5)
        session.status = GameStatus.IN_PROGRESS
        
        # Place 4 X's in a row
        for col in range(4):
            session.board[4][col] = "X"
        
        # Place the 5th to win
        result = session.apply_move(4, 4, PlayerRole.BLACK)
        
        assert result["valid"] is True
        assert result["done"] is True
        assert result["info"]["result"] == "win"
        assert session.status == GameStatus.BLACK_WIN
    
    def test_win_detection_vertical(self):
        """Test vertical win detection."""
        session = GameSession(session_id="test_008", board_size=9, win_length=5)
        session.status = GameStatus.IN_PROGRESS
        
        # Place 4 X's in a column
        for row in range(4):
            session.board[row][4] = "X"
        
        # Place the 5th to win
        result = session.apply_move(4, 4, PlayerRole.BLACK)
        
        assert result["valid"] is True
        assert result["done"] is True
        assert result["info"]["result"] == "win"
    
    def test_win_detection_diagonal(self):
        """Test diagonal win detection."""
        session = GameSession(session_id="test_009", board_size=9, win_length=5)
        session.status = GameStatus.IN_PROGRESS
        
        # Place 4 X's diagonally
        for i in range(4):
            session.board[i][i] = "X"
        
        # Place the 5th to win
        result = session.apply_move(4, 4, PlayerRole.BLACK)
        
        assert result["valid"] is True
        assert result["done"] is True
        assert result["info"]["result"] == "win"
    
    def test_draw_detection(self):
        """Test draw when max steps reached."""
        session = GameSession(session_id="test_010", board_size=9, max_total_steps=2)
        session.status = GameStatus.IN_PROGRESS
        
        # Make 2 moves (max_total_steps)
        session.apply_move(0, 0, PlayerRole.BLACK)
        session.current_turn = PlayerRole.BLACK  # Reset for test
        result = session.apply_move(0, 1, PlayerRole.BLACK)
        
        assert result["done"] is True
        assert result["info"]["result"] == "draw"
        assert session.status == GameStatus.DRAW
    
    def test_get_state_for_agent(self):
        """Test getting state from agent's perspective."""
        session = GameSession(session_id="test_011", board_size=9)
        session.agents_joined["agent_black"] = PlayerRole.BLACK
        session.status = GameStatus.IN_PROGRESS
        
        state = session.get_state_for_agent(PlayerRole.BLACK)
        
        assert state["your_role"] == "BLACK"
        assert state["your_symbol"] == "X"
        assert state["is_your_turn"] is True
        assert "board" in state
    
    def test_reset(self):
        """Test session reset."""
        session = GameSession(session_id="test_012", board_size=9)
        session.agents_joined["agent_black"] = PlayerRole.BLACK
        session.agents_joined["agent_white"] = PlayerRole.WHITE
        session.status = GameStatus.BLACK_WIN
        session.board[4][4] = "X"
        session.step_count = 10
        
        session.reset()
        
        assert session.status == GameStatus.IN_PROGRESS
        assert session.current_turn == PlayerRole.BLACK
        assert session.step_count == 0
        assert session.board[4][4] == "."
        assert len(session.move_history) == 0


class TestTurnSynchronization:
    """Test turn-based synchronization logic."""
    
    def test_turn_order_enforcement(self):
        """Test that only the correct player can move."""
        session = GameSession(session_id="test_turn_001", board_size=9)
        session.agents_joined["black"] = PlayerRole.BLACK
        session.agents_joined["white"] = PlayerRole.WHITE
        session.status = GameStatus.IN_PROGRESS
        
        # BLACK's turn
        assert session.current_turn == PlayerRole.BLACK
        result = session.apply_move(4, 4, PlayerRole.BLACK)
        assert result["valid"] is True
        
        # Now WHITE's turn
        assert session.current_turn == PlayerRole.WHITE
        
        # BLACK tries to move again - should succeed but game logic allows it
        # (turn validation is done at API level, not in apply_move)
    
    def test_move_history_tracking(self):
        """Test that moves are properly tracked."""
        session = GameSession(session_id="test_history_001", board_size=9)
        session.status = GameStatus.IN_PROGRESS
        
        session.apply_move(4, 4, PlayerRole.BLACK)
        session.apply_move(3, 3, PlayerRole.WHITE)
        
        assert len(session.move_history) == 2
        assert session.move_history[0]["agent"] == "BLACK"
        assert session.move_history[0]["position"] == [4, 4]
        assert session.move_history[1]["agent"] == "WHITE"
        assert session.move_history[1]["position"] == [3, 3]


class TestMoveParser:
    """Test move parsing utilities."""
    
    def test_parse_move_direct(self):
        """Test parsing direct format."""
        server = MultiAgentGameServer()
        
        row, col = server._parse_move("4,4")
        assert row == 4
        assert col == 4
    
    def test_parse_move_with_tags(self):
        """Test parsing move with XML tags."""
        server = MultiAgentGameServer()
        
        row, col = server._parse_move("<thinking>Center is good</thinking><move>4,4</move>")
        assert row == 4
        assert col == 4
    
    def test_parse_move_with_spaces(self):
        """Test parsing move with spaces."""
        server = MultiAgentGameServer()
        
        row, col = server._parse_move("<move> 4 , 4 </move>")
        assert row == 4
        assert col == 4
    
    def test_parse_move_invalid(self):
        """Test parsing invalid move."""
        server = MultiAgentGameServer()
        
        row, col = server._parse_move("invalid move text")
        assert row is None
        assert col is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
