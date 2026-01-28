#!/usr/bin/env python3
"""Rule-based Gomoku Interaction for Validation.

This module provides an interaction class where the LLM plays against
a rule-based Gomoku opponent. Used for validation to provide a consistent
benchmark without requiring coordination between two LLM agents.

The LLM always plays as BLACK (X), and the rule-based opponent plays as WHITE (O).
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from verl.interactions.base import BaseInteraction
from opentinker.environment.gomoku.gomoku_game import GomokuGame

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RuleBasedGomokuInteraction(BaseInteraction):
    """Interaction class for LLM vs rule-based Gomoku opponent.
    
    This is used during validation to provide a consistent benchmark.
    The LLM plays as BLACK (X), rule-based opponent plays as WHITE (O).
    
    The rule-based opponent uses a strategic algorithm that:
    1. Tries to find winning moves
    2. Blocks opponent's winning moves
    3. Uses position scoring with center preference
    4. Has configurable randomness for variety
    """

    # Reward structure (matching MultiAgentGomokuInteraction)
    REWARD_WIN = 1.0
    REWARD_LOSS = -1.0
    REWARD_DRAW = -0.1
    REWARD_INVALID_MOVE = -0.5
    REWARD_ONGOING = 0.0

    def __init__(self, config: dict):
        """Initialize the rule-based Gomoku interaction.
        
        Args:
            config: Configuration dictionary with:
                - board_size: Size of the board (default: 9)
                - win_length: Number in a row to win (default: 5)
                - max_total_steps: Maximum steps before timeout (default: 81)
                - max_invalid_moves: Max invalid moves before game over (default: 3)
        """
        self.board_size = config.get("board_size", 9)
        self.win_length = config.get("win_length", 5)
        self.max_total_steps = config.get("max_total_steps", 81)
        self.max_invalid_moves = config.get("max_invalid_moves", 3)
        
        # Instance tracking
        self._instance_dict: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            f"RuleBasedGomokuInteraction initialized: "
            f"board_size={self.board_size}, max_steps={self.max_total_steps}"
        )

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> str:
        """Initialize a new game instance.
        
        Args:
            instance_id: Unique identifier for this game instance
            **kwargs: Additional arguments (env_kwargs passed to game reset)
            
        Returns:
            The instance_id
        """
        # Use deterministic game_session_id if provided (matching multi-agent training)
        game_session_id = kwargs.get("game_session_id")
        request_id = kwargs.get("request_id")
        
        if game_session_id:
            instance_id = game_session_id
        elif instance_id is None:
            # Fallback for manual testing: use request_id or a fixed base
            instance_id = request_id or f"local_game_{hash(self) % 1000}"
        
        # Determine roles
        agent_role = kwargs.get("agent_role", "BLACK").upper()
        # GomokuGame uses agent_role to determine prompts. 
        # In single-agent mode (against AI), GomokuGame assumes LLM is X.
        # But we want to preserve the perspective (BLACK=first, WHITE=second).
        
        # Create a new GomokuGame instance
        game = GomokuGame(
            board_size=self.board_size,
            win_length=self.win_length,
            max_total_steps=self.max_total_steps,
            agent_role=agent_role,  # Preserve the original role for prompts
        )
        
        # Get initial moves from kwargs if provided
        env_kwargs = kwargs.get("env_kwargs", {})
        initial_moves = env_kwargs.get("initial_moves", None)
        
        # Reset the game
        initial_observation = game.reset(initial_moves=initial_moves)
        
        # If agent is WHITE, robot (BLACK) must move first
        if agent_role == "WHITE":
            logger.info(f"[{instance_id[:8]}] Agent is WHITE, robot moves first")
            # Make the first move for the robot (env_symbol='X' in this case)
            env_row, env_col = game._make_env_move()
            game.board[env_row][env_col] = game.env_symbol
            game.move_count += 1
            game.last_move = (env_row, env_col)
            
            # The observation should now show the board with X (BLACK) move.
            # And it should prompt the LLM to move as O (WHITE).
            initial_observation = (
                f"Opponent (BLACK/X) moved to ({env_row},{env_col}).\n\n"
                f"{game._render_board()}\n\n"
                f"Your turn (O). Provide your thinking and move:"
            )
        
        # Store instance data
        self._instance_dict[instance_id] = {
            "game": game,
            "step_count": 0,
            "invalid_move_count": 0,
            "cumulative_reward": 0.0,
            "game_ended": False,
            "initial_board_state": initial_observation,
            "agent_role": agent_role,
        }
        
        logger.info(f"[{instance_id[:8]}] Started rule-based Gomoku game")
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """Execute LLM's move and get rule-based opponent's response.
        
        Args:
            instance_id: The game instance ID
            messages: Conversation history, last message should be LLM's move
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (should_terminate, observation, reward, info)
        """
        if instance_id not in self._instance_dict:
            return (True, "Error: Game instance not found", 0.0, {"error": "instance_not_found"})
        
        instance = self._instance_dict[instance_id]
        game: GomokuGame = instance["game"]
        
        if instance["game_ended"]:
            return (True, "Game already ended", 0.0, {"error": "game_already_ended"})
        
        # Extract action from the last assistant message
        action = self._extract_action(messages)
        if action is None:
            return (True, "Error: No action found in messages", 0.0, {"error": "no_action"})
        
        # Execute the step
        result = game.step(action)
        instance["step_count"] += 1
        
        # Process the result
        observation = result.observation
        reward = result.reward
        instance["cumulative_reward"] += reward
        done = result.done
        info = result.info
        
        # Check for invalid move
        if info.get("invalid_move", False):
            instance["invalid_move_count"] += 1
            
            # Check if too many invalid moves
            if instance["invalid_move_count"] >= self.max_invalid_moves:
                instance["game_ended"] = True
                return (
                    True,
                    f"Game Over - Too many invalid moves ({self.max_invalid_moves})\n{observation}",
                    self.REWARD_INVALID_MOVE,
                    {"game_result": "invalid_moves", "invalid_count": instance["invalid_move_count"]},
                )
            
            # Allow retry with penalty (skip_assistant_message to exclude failed attempt)
            return (
                False,  # Don't terminate - allow retry
                f"Invalid move: {info.get('error', 'Unknown error')}. Please try again.\n{observation}",
                self.REWARD_INVALID_MOVE,
                {"skip_assistant_message": True, "invalid_move": True},
            )
        
        # Valid move - reset invalid count
        instance["invalid_move_count"] = 0
        
        if done:
            instance["game_ended"] = True
            game_result = info.get("result", "unknown")
            
            # Determine final reward
            winner = info.get("winner")
            if winner == game.llm_symbol:
                final_reward = self.REWARD_WIN
            elif winner == game.env_symbol:
                final_reward = self.REWARD_LOSS
            else:  # draw or timeout
                final_reward = self.REWARD_DRAW
            
            logger.info(
                f"[{instance_id[:8]}] Game ended: {game_result}, "
                f"reward={final_reward}, steps={instance['step_count']}"
            )
            
            return (
                True,
                f"Game Over - {game_result.upper()}\n{observation}",
                final_reward,
                {"game_result": game_result, "steps": instance["step_count"]},
            )
        
        # Game continues - rule-based opponent has already moved in game.step()
        return (
            False,
            observation,
            self.REWARD_ONGOING,
            {"steps": instance["step_count"]},
        )

    def _extract_action(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the action from the last assistant message.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            The action string or None if not found
        """
        if not messages:
            return None
        
        # Find the last assistant message
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content:
                    return content
        
        return None

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Calculate the final score for this game.
        
        Args:
            instance_id: The game instance ID
            
        Returns:
            The cumulative reward for the game
        """
        if instance_id not in self._instance_dict:
            return 0.0
        
        instance = self._instance_dict[instance_id]
        return instance.get("cumulative_reward", 0.0)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Clean up resources for this game instance.
        
        Args:
            instance_id: The game instance ID
        """
        if instance_id in self._instance_dict:
            logger.info(f"[{instance_id[:8]}] Finalized rule-based Gomoku game")
            del self._instance_dict[instance_id]
