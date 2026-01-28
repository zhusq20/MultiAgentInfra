#!/usr/bin/env python3
"""Gomoku Game Implementation.

This module provides the GomokuGame class that implements AbstractGame interface.
It contains only the core game logic - the HTTP server and data generation are
handled by the generic base classes.

Example:
    from gomoku_game import GomokuGame

    game = GomokuGame(board_size=9)
    obs = game.reset()
    result = game.step("<thinking>...</thinking><move>4,4</move>")
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from opentinker.environment.base_game import AbstractGame, StepResult


class GomokuGame(AbstractGame):
    """Gomoku (Five in a Row) game implementation.

    This implementation:
    - LLM plays as X, environment plays as O
    - Supports configurable board size and win length
    - Includes strategic AI opponent with configurable difficulty
    - Provides reward shaping for training

    Attributes:
        board_size: Size of the board (default: 9)
        win_length: Number in a row to win (default: 5)
        max_total_steps: Maximum steps before timeout
    """

    # Reward constants - SPARSE REWARDS + FORMAT REWARDS
    REWARD_WIN = 10.0
    REWARD_LOSS = -10.0
    REWARD_DRAW = 0.0
    REWARD_VALID_MOVE = 0.0
    REWARD_INVALID_FORMAT = -1.0
    REWARD_INVALID_POSITION = -1.0

    # Invalid move limits
    MAX_CONSECUTIVE_INVALID_MOVES = 100
    REWARD_INVALID_GAME_OVER = -20.0

    # Timeout
    DEFAULT_MAX_TOTAL_STEPS = 40
    REWARD_TIMEOUT = -20.0

    # Opponent difficulty (0.0 = smart, 1.0 = random)
    OPPONENT_RANDOM_PROB = 0.3

    # Symbols
    EMPTY = "."
    PLAYER_X = "X"  # LLM plays as X
    PLAYER_O = "O"  # Environment plays as O

    def __init__(
        self,
        board_size: int = 9,
        win_length: int = 5,
        max_total_steps: Optional[int] = None,
        max_initial_moves: Optional[int] = None,
        empty_board_prob: Optional[float] = None,
        agent_role: Optional[str] = None,
    ):
        """Initialize Gomoku game.

        Args:
            board_size: Size of the board
            win_length: Number in a row to win
            max_total_steps: Maximum steps before timeout
            max_initial_moves: Maximum initial moves to place before game starts
            empty_board_prob: Probability of starting with an empty board
            agent_role: For multi-agent mode: "BLACK" (X, first) or "WHITE" (O, second).
                       If None, uses single-agent mode where LLM always plays as X.
        """
        self.board_size = board_size
        self.win_length = min(win_length, board_size)
        self.max_total_steps = max_total_steps or self.DEFAULT_MAX_TOTAL_STEPS
        # Use provided values or fall back to class defaults
        self.max_initial_moves = (
            max_initial_moves
            if max_initial_moves is not None
            else self.MAX_INITIAL_MOVES
        )
        self.empty_board_prob = (
            empty_board_prob if empty_board_prob is not None else self.EMPTY_BOARD_PROB
        )
        # Multi-agent mode: agent_role determines the player's role
        # None means single-agent mode (LLM plays as X against AI)
        self.agent_role = agent_role.upper() if agent_role else None
        
        # Determine symbols based on role
        if self.agent_role == "WHITE":
            self.llm_symbol = self.PLAYER_O
            self.env_symbol = self.PLAYER_X
        else:
            self.llm_symbol = self.PLAYER_X
            self.env_symbol = self.PLAYER_O
            
        self._init_game_state()

    def _init_game_state(self):
        """Initialize/reset game state variables."""
        self.board = [[self.EMPTY] * self.board_size for _ in range(self.board_size)]
        self.current_player = self.PLAYER_X
        self.move_count = 0
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.consecutive_invalid_moves = 0
        self.total_step_count = 0

    def reset(self, initial_moves: Optional[List] = None, **kwargs) -> str:
        """Reset the game and optionally apply initial moves.

        Args:
            initial_moves: List of [row, col] moves to apply
            **kwargs: Ignored (for compatibility)

        Returns:
            Text representation of the board state
        """
        self._init_game_state()

        # Apply initial moves if provided
        if initial_moves:
            for move in initial_moves:
                if len(move) >= 2:
                    row, col = move[0], move[1]
                    if (
                        self._is_valid_position(row, col)
                        and self.board[row][col] == self.EMPTY
                    ):
                        self.board[row][col] = self.current_player
                        self.move_count += 1
                        self._toggle_player()

        return self._render_board()

    def step(self, action: str) -> StepResult:
        """Execute a move and return the result.

        Args:
            action: String containing the move in <move>row,col</move> format

        Returns:
            StepResult with observation, reward, done flag, and info
        """
        if self.game_over:
            return StepResult(
                observation=self._render_board(),
                reward=-5.0,
                done=True,
                info={"error": "Game already over"},
            )

        # Increment total step count
        self.total_step_count += 1

        # Check for timeout
        if self.total_step_count >= self.max_total_steps:
            self.game_over = True
            return StepResult(
                observation=f"TIMEOUT: Maximum steps ({self.max_total_steps}) reached.\n\n{self._render_board()}",
                reward=self.REWARD_TIMEOUT,
                done=True,
                info={"error": "timeout", "total_steps": self.total_step_count},
            )

        # Parse the action
        row, col = self._parse_action(action)

        # Handle invalid moves
        invalid_result = self._handle_invalid_move(row, col, action)
        if invalid_result is not None:
            return invalid_result

        # Valid move - reset consecutive invalid counter
        self.consecutive_invalid_moves = 0

        # Make the move
        self.board[row][col] = self.llm_symbol
        self.move_count += 1
        self.last_move = (row, col)

        # Check for win
        if self._check_win(row, col, self.llm_symbol):
            self.game_over = True
            self.winner = self.llm_symbol
            return StepResult(
                observation=f"Congratulations! You win!\n\n{self._render_board()}",
                reward=self.REWARD_WIN,
                done=True,
                info={
                    "winner": self.llm_symbol,
                    "move": [row, col],
                    "board_state": self.get_state(),
                },
            )

        # Check for draw
        if self.move_count >= self.board_size * self.board_size:
            self.game_over = True
            return StepResult(
                observation=f"Game is a draw!\n\n{self._render_board()}",
                reward=self.REWARD_DRAW,
                done=True,
                info={"winner": None, "draw": True},
            )

        # Environment's turn
        env_row, env_col = self._make_env_move()
        self.board[env_row][env_col] = self.env_symbol
        self.move_count += 1

        # Check if environment wins
        if self._check_win(env_row, env_col, self.env_symbol):
            self.game_over = True
            self.winner = self.env_symbol
            return StepResult(
                observation=f"You lose! Opponent placed {self.env_symbol} at ({env_row},{env_col}).\n\n{self._render_board()}",
                reward=self.REWARD_LOSS,
                done=True,
                info={"winner": self.env_symbol, "env_move": [env_row, env_col]},
            )

        # Check for draw after environment move
        if self.move_count >= self.board_size * self.board_size:
            self.game_over = True
            return StepResult(
                observation=f"Game is a draw!\n\n{self._render_board()}",
                reward=self.REWARD_DRAW,
                done=True,
                info={"winner": None, "draw": True},
            )

        # Game continues
        observation = (
            f"Opponent placed {self.env_symbol} at ({env_row},{env_col}).\n\n"
            f"{self._render_board()}\n\n"
            f"Your turn ({self.llm_symbol}). Analyze and provide your move:\n"
            f"<thinking>your analysis</thinking>\n"
            f"<move>row,col</move>"
        )

        return StepResult(
            observation=observation,
            reward=self.REWARD_VALID_MOVE,
            done=False,
            info={"your_move": [row, col], "env_move": [env_row, env_col]},
        )

    def get_system_prompt(self) -> str:
        """Return the system prompt for Gomoku."""
        center = self.board_size // 2
        
        # Multi-agent mode: role-aware prompts
        if self.agent_role == "BLACK":
            symbol = "X"
            role_info = "You play first."
        elif self.agent_role == "WHITE":
            symbol = "O"
            role_info = "You play second (after opponent's first move)."
        else:
            # Single-agent mode: LLM always plays as X
            symbol = "X"
            role_info = ""
        
        role_line = f"You play as {self.agent_role} ({symbol}). {role_info}\n" if self.agent_role else f"You play as {symbol}.\n"
        
        return (
            f"You are playing Gomoku (Five in a Row) on a {self.board_size}x{self.board_size} board.\n"
            f"{role_line}"
            f"Your goal is to get {self.win_length} in a row.\n\n"
            f"IMPORTANT: You MUST respond in the following format:\n"
            f"1. First, analyze the board in <thinking></thinking> tags\n"
            f"2. Then, output your move in <move>row,col</move> tags\n\n"
            f"Example: <thinking>I should take the center.</thinking><move>{center},{center}</move>"
        )

    def get_initial_user_message(self) -> str:
        """Return the initial user message for Gomoku."""
        # Multi-agent mode: role-aware initial message
        if self.agent_role == "BLACK":
            return (
                f"Game starts. Here's the empty board:\n\n"
                f"{self._render_board()}\n\n"
                f"You play first (X). Analyze and provide your move:"
            )
        elif self.agent_role == "WHITE":
            return (
                f"Game started. Waiting for opponent's first move.\n\n"
                f"{self._render_board()}\n\n"
                f"After opponent moves, provide your thinking and move:"
            )
        else:
            # Single-agent mode
            return (
                f"Game starts. Here's the board:\n\n"
                f"{self._render_board()}\n\n"
                f"Your turn (X). Provide your thinking and move:"
            )

    def get_state(self) -> Dict[str, Any]:
        """Return current game state."""
        return {
            "board_array": [row[:] for row in self.board],
            "board_visual": self._render_board(),
            "move_count": self.move_count,
            "last_move": list(self.last_move) if self.last_move else None,
            "game_over": self.game_over,
            "winner": self.winner,
        }

    # =========================================================================
    # Data Generation Methods (for training)
    # =========================================================================

    # Configuration for data generation
    MAX_INITIAL_MOVES = 6
    EMPTY_BOARD_PROB = 0.2

    def generate_initial_state(self) -> Dict[str, Any]:
        """Generate random initial state for training data.

        Returns a dict with 'initial_moves' that reset() will use.
        """
        if random.random() < self.empty_board_prob:
            return {"initial_moves": []}

        # Generate random valid moves
        num_moves = random.randint(0, self.max_initial_moves)
        if num_moves == 0:
            return {"initial_moves": []}

        all_positions = [
            (r, c) for r in range(self.board_size) for c in range(self.board_size)
        ]
        selected = random.sample(all_positions, min(num_moves, len(all_positions)))

        # Filter to ensure no winner
        moves = []
        board = [["" for _ in range(self.board_size)] for _ in range(self.board_size)]

        for i, (r, c) in enumerate(selected):
            player = "X" if i % 2 == 0 else "O"
            board[r][c] = player

            # Check if this creates a win
            if not self._check_win_on_board(board, r, c, player):
                moves.append([r, c])
            else:
                board[r][c] = ""  # Undo

        return {"initial_moves": moves}

    def _check_win_on_board(
        self, board: List[List[str]], row: int, col: int, player: str
    ) -> bool:
        """Check win on a given board (for data generation)."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1

            r, c = row + dr, col + dc
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and board[r][c] == player
            ):
                count += 1
                r += dr
                c += dc

            r, c = row - dr, col - dc
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and board[r][c] == player
            ):
                count += 1
                r -= dr
                c -= dc

            if count >= self.win_length:
                return True

        return False

    def get_user_message_with_state(
        self, initial_moves: Optional[List] = None, **kwargs
    ) -> str:
        """Generate user message with rendered initial state for prompt."""
        # Temporarily apply moves to render the board
        saved_board = [row[:] for row in self.board]
        saved_count = self.move_count

        self.board = [[self.EMPTY] * self.board_size for _ in range(self.board_size)]
        if initial_moves:
            for i, move in enumerate(initial_moves):
                if len(move) >= 2:
                    r, c = move[0], move[1]
                    if 0 <= r < self.board_size and 0 <= c < self.board_size:
                        player = "X" if i % 2 == 0 else "O"
                        self.board[r][c] = player

        board_str = self._render_board()

        # Restore board
        self.board = saved_board
        self.move_count = saved_count

        symbol = "X" if self.agent_role != "WHITE" else "O"
        if initial_moves:
            return f"Current board state:\n\n{board_str}\n\nYour turn ({symbol}). Analyze and provide your move:"
        else:
            return f"Game starts. Here's the empty board:\n\n{board_str}\n\nYou play first ({symbol}). Analyze and provide your move:"

    def get_interaction_name(self) -> str:
        """Return interaction name for Gomoku."""
        return "gomoku"

    # =====================
    # Private helper methods
    # =====================

    def _parse_action(self, action: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse action string to extract row and column."""
        action = action.strip()
        match = re.search(
            r"<move>\s*(\d+)\s*,\s*(\d+)\s*</move>", action, re.IGNORECASE
        )
        if match:
            try:
                return int(match.group(1)), int(match.group(2))
            except ValueError:
                pass
        return None, None

    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within board bounds."""
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def _toggle_player(self):
        """Switch current player."""
        self.current_player = (
            self.PLAYER_O if self.current_player == self.PLAYER_X else self.PLAYER_X
        )

    def _handle_invalid_move(
        self, row: Optional[int], col: Optional[int], action: str
    ) -> Optional[StepResult]:
        """Handle invalid move cases. Returns StepResult if invalid, None if valid."""
        # Invalid format
        if row is None or col is None:
            self.consecutive_invalid_moves += 1
            if self.consecutive_invalid_moves >= self.MAX_CONSECUTIVE_INVALID_MOVES:
                self.game_over = True
                return StepResult(
                    observation="ERROR: Too many invalid moves. Game over.",
                    reward=self.REWARD_INVALID_GAME_OVER,
                    done=True,
                    info={"error": "max_invalid_moves"},
                )
            return StepResult(
                observation=(
                    "ERROR: Invalid move format.\n"
                    "Use: <thinking>analysis</thinking><move>row,col</move>"
                ),
                reward=self.REWARD_INVALID_FORMAT,
                done=False,
                info={"error": "invalid_format"},
            )

        # Out of bounds
        if not self._is_valid_position(row, col):
            self.consecutive_invalid_moves += 1
            if self.consecutive_invalid_moves >= self.MAX_CONSECUTIVE_INVALID_MOVES:
                self.game_over = True
                return StepResult(
                    observation="ERROR: Too many invalid moves. Game over.",
                    reward=self.REWARD_INVALID_GAME_OVER,
                    done=True,
                    info={"error": "max_invalid_moves"},
                )
            return StepResult(
                observation=(
                    f"ERROR: Position ({row},{col}) is OUT OF BOUNDS.\n"
                    f"Valid range: 0-{self.board_size-1}\n\n{self._render_board()}"
                ),
                reward=self.REWARD_INVALID_POSITION,
                done=False,
                info={"error": "out_of_bounds", "row": row, "col": col},
            )

        # Position occupied
        if self.board[row][col] != self.EMPTY:
            self.consecutive_invalid_moves += 1
            if self.consecutive_invalid_moves >= self.MAX_CONSECUTIVE_INVALID_MOVES:
                self.game_over = True
                return StepResult(
                    observation="ERROR: Too many invalid moves. Game over.",
                    reward=self.REWARD_INVALID_GAME_OVER,
                    done=True,
                    info={"error": "max_invalid_moves"},
                )
            return StepResult(
                observation=(
                    f"ERROR: Position ({row},{col}) is OCCUPIED.\n\n{self._render_board()}"
                ),
                reward=self.REWARD_INVALID_POSITION,
                done=False,
                info={"error": "occupied", "row": row, "col": col},
            )

        return None  # Move is valid

    def _check_win(self, row: int, col: int, player: str) -> bool:
        """Check if the player has won after placing at (row, col)."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1

            # Positive direction
            r, c = row + dr, col + dc
            while self._is_valid_position(r, c) and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc

            # Negative direction
            r, c = row - dr, col - dc
            while self._is_valid_position(r, c) and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= self.win_length:
                return True

        return False

    def _count_in_direction(
        self, row: int, col: int, dr: int, dc: int, player: str
    ) -> int:
        """Count consecutive pieces in a direction."""
        count = 0
        r, c = row + dr, col + dc
        while self._is_valid_position(r, c) and self.board[r][c] == player:
            count += 1
            r += dr
            c += dc
        return count

    def _make_env_move(self) -> Tuple[int, int]:
        """Make a strategic move for the environment (O)."""
        empty_cells = [
            (r, c)
            for r in range(self.board_size)
            for c in range(self.board_size)
            if self.board[r][c] == self.EMPTY
        ]

        if not empty_cells:
            return empty_cells[0]  # Should not happen

        # Random move with configured probability
        if random.random() < self.OPPONENT_RANDOM_PROB:
            return random.choice(empty_cells)

        # Check for winning move
        for r, c in empty_cells:
            self.board[r][c] = self.env_symbol
            if self._check_win(r, c, self.env_symbol):
                self.board[r][c] = self.EMPTY
                return r, c
            self.board[r][c] = self.EMPTY

        # Block opponent's winning move
        for r, c in empty_cells:
            self.board[r][c] = self.llm_symbol
            if self._check_win(r, c, self.llm_symbol):
                self.board[r][c] = self.EMPTY
                return r, c
            self.board[r][c] = self.EMPTY

        # Find best move by scoring
        best_move = None
        best_score = -1
        for r, c in empty_cells:
            score = 0
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                count = self._count_in_direction(r, c, dr, dc, self.env_symbol)
                count += self._count_in_direction(r, c, -dr, -dc, self.env_symbol)
                score += count

            # Center preference
            center = self.board_size // 2
            score += 0.1 / (abs(r - center) + abs(c - center) + 1)

            if score > best_score:
                best_score = score
                best_move = (r, c)

        if best_move and best_score > 0:
            return best_move

        # Random move with center preference
        center = self.board_size // 2
        empty_cells.sort(key=lambda x: abs(x[0] - center) + abs(x[1] - center))
        top_choices = empty_cells[: max(1, len(empty_cells) // 3)]
        return random.choice(top_choices)

    def _render_board(self) -> str:
        """Render the board as a text string."""
        lines = []
        header = "  " + " ".join(str(c) for c in range(self.board_size))
        lines.append(header)

        for r in range(self.board_size):
            row_str = f"{r} " + " ".join(self.board[r])
            lines.append(row_str)

        return "\n".join(lines)
