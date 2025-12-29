"""Gomoku Environment Module for LLM Training.

Usage:
    from opentinker.environment.base_game_environment import GameEnvironment
    from opentinker.environment.gomoku import GomokuGame
    from opentinker.environment.game_stats_client import GameStatsClient

    env = GameEnvironment(game_class=GomokuGame, config=config)
    stats_client = GameStatsClient(env_endpoint)

    # Optional: GomokuGameStats for server-side metrics
    from opentinker.environment.gomoku import GomokuGameStats  # may be None
    
    # Multi-agent components
    from opentinker.environment.gomoku import MultiAgentGameServer
    from opentinker.environment.gomoku import MultiAgentGomokuInteraction
"""

from .gomoku_game import GomokuGame

# GomokuGameStats is optional - only available if gomoku_stats.py exists
try:
    from .gomoku_stats import GomokuGameStats
except ImportError:
    GomokuGameStats = None


# Lazy imports for multi-agent components to avoid RuntimeWarning
# when running with python -m
_multi_agent_cache = {}


def __getattr__(name):
    """Lazy import for multi-agent components."""
    if name in ("MultiAgentGameServer", "GameSession", "PlayerRole"):
        if "multi_agent_game_server" not in _multi_agent_cache:
            try:
                from . import multi_agent_game_server as _module
                _multi_agent_cache["multi_agent_game_server"] = _module
                _multi_agent_cache["MultiAgentGameServer"] = _module.MultiAgentGameServer
                _multi_agent_cache["GameSession"] = _module.GameSession
                _multi_agent_cache["PlayerRole"] = _module.PlayerRole
            except ImportError:
                _multi_agent_cache["MultiAgentGameServer"] = None
                _multi_agent_cache["GameSession"] = None
                _multi_agent_cache["PlayerRole"] = None
        return _multi_agent_cache.get(name)
    
    if name == "MultiAgentGomokuInteraction":
        if "MultiAgentGomokuInteraction" not in _multi_agent_cache:
            try:
                from .multi_agent_gomoku_interaction import MultiAgentGomokuInteraction as _cls
                _multi_agent_cache["MultiAgentGomokuInteraction"] = _cls
            except ImportError:
                _multi_agent_cache["MultiAgentGomokuInteraction"] = None
        return _multi_agent_cache.get(name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GomokuGame",
    "GomokuGameStats",
    "MultiAgentGameServer",
    "GameSession",
    "PlayerRole",
    "MultiAgentGomokuInteraction",
]
