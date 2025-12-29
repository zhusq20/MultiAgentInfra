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
"""Dual Agent Gomoku Training Client.

This script enables training two LLM agents to play Gomoku against each other.
Each agent runs as a separate process with its own PPO training server.

Usage:
    # Terminal 1: Start game server
    python -m opentinker.environment.gomoku.multi_agent_game_server --port 8790
    
    # Terminal 2: Start phase coordinator
    python -m opentinker.server.phase_coordinator --port 8791
    
    # Terminal 3: Start BLACK agent
    AGENT_ROLE=BLACK python opentinker/client/dual_agent_gomoku_rl.py \
        scheduler_url=http://localhost:8780 \
        session_id=game_001
    
    # Terminal 4: Start WHITE agent
    AGENT_ROLE=WHITE python opentinker/client/dual_agent_gomoku_rl.py \
        scheduler_url=http://localhost:8780 \
        session_id=game_001
"""

import atexit
import os
import sys
import time
import tempfile
import signal
import requests
from typing import Optional

from omegaconf import OmegaConf
import hydra

from utils.http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.base_game_environment import GameEnvironment
from opentinker.environment.gomoku import GomokuGame
from opentinker.environment.game_stats_client import GameStatsClient
from utils.utils import resolve_paths_in_config
from utils.scheduler_client_lifecycle import get_lifecycle_manager
from opentinker.client.utils.phase_coordinator_client import PhaseCoordinatorClient


def create_multi_agent_interaction_config(
    game_server_url: str,
    agent_role: str,
    session_id: str = "default_session",
    board_size: int = 9,
    max_total_steps: int = 40,
) -> str:
    """Create a temporary interaction config file for multi-agent Gomoku.
    
    Returns:
        Path to the temporary config file.
    """
    # Note: interaction_registry.py expects 'class_name' not 'class_path'
    config_content = f"""# Multi-Agent Gomoku Interaction Configuration
# Generated for {agent_role} agent

interaction:
  - name: multi_agent_gomoku
    class_name: opentinker.environment.gomoku.multi_agent_gomoku_interaction.MultiAgentGomokuInteraction
    config:
      game_server_url: {game_server_url}
      agent_role: {agent_role}
      session_id: {session_id}
      board_size: {board_size}
      max_total_steps: {max_total_steps}
      timeout: 300
"""
    
    # Create temp file
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix=f"multi_agent_interaction_{agent_role}_")
    with os.fdopen(fd, "w") as f:
        f.write(config_content)
    
    return path


# Global variables for cleanup
_cleanup_info = {
    "game_server_url": None,
    "coordinator_url": None,
    "session_id": None,
    "agent_role": None,
    "cleaned_up": False,
}


def cleanup_game_session():
    """Clean up the game session on the game server and phase coordinator.
    
    This function is called when the client is killed or terminates to
    notify the game server and phase coordinator to delete the corresponding session.
    """
    if _cleanup_info["cleaned_up"]:
        return
    
    _cleanup_info["cleaned_up"] = True
    
    game_server_url = _cleanup_info.get("game_server_url")
    coordinator_url = _cleanup_info.get("coordinator_url")
    session_id = _cleanup_info.get("session_id")
    agent_role = _cleanup_info.get("agent_role")
    
    if not session_id:
        return
    
    print(f"\n⚠ Cleaning up session {session_id} ({agent_role})...")
    
    # Clean up game server
    if game_server_url:
        try:
            response = requests.delete(
                f"{game_server_url}/session/{session_id}",
                timeout=5,
            )
            if response.status_code == 200:
                print(f"✓ Game session {session_id} deleted from game server")
            else:
                print(f"⚠ Failed to delete game session: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"⚠ Error cleaning up game session: {e}")
    
    # Clean up phase coordinator
    if coordinator_url:
        try:
            response = requests.delete(
                f"{coordinator_url}/session/{session_id}",
                timeout=5,
            )
            if response.status_code == 200:
                print(f"✓ Coordinator session {session_id} deleted from phase coordinator")
            else:
                print(f"⚠ Failed to delete coordinator session: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"⚠ Error cleaning up coordinator session: {e}")


def signal_handler(signum, frame):
    """Handle termination signals."""
    sig_name = signal.Signals(signum).name
    print(f"\n⚠ Received signal {sig_name}, cleaning up...")
    cleanup_game_session()
    sys.exit(1)


@hydra.main(config_path="client_config", config_name="dual_gomoku_param.yaml")
def main(args):
    """Main training function for dual agent Gomoku."""
    # Resolve paths
    args = resolve_paths_in_config(args)
    
    # Get lifecycle manager
    lifecycle = get_lifecycle_manager()
    
    # Determine agent role from environment or config
    agent_role = os.environ.get("AGENT_ROLE", args.get("agent_role", "BLACK")).upper()
    session_id = os.environ.get("SESSION_ID", args.get("session_id", "default_session"))
    agent_id = f"agent_{agent_role}"
    
    print("=" * 60)
    print(f"Dual Agent Gomoku Training - {agent_role}")
    print("=" * 60)
    print(f"  Session ID: {session_id}")
    print(f"  Agent ID: {agent_id}")
    print(f"  Agent Role: {agent_role}")
    
    # Initialize tracing (optional)
    enable_tracing = args.get("enable_tracing", False)
    if enable_tracing:
        try:
            from opentinker.utils.rollout_trace_saver import init_weave_tracing
            
            weave_project = args.get("weave_project", "dual-gomoku-training")
            init_weave_tracing(
                project_name=weave_project,
                experiment_name=f"{args.experiment_name}_{agent_role}",
                token2text=True,
            )
        except Exception as e:
            print(f"⚠ Failed to initialize Weave tracing: {e}")
    
    # Get service URLs
    game_server_url = args.get("game_server_url", "http://localhost:8790")
    coordinator_url = args.get("coordinator_url", "http://localhost:8791")
    scheduler_url = args.get("scheduler_url", "http://localhost:8780")
    scheduler_api_key = args.get("scheduler_api_key", None)
    
    print(f"\nService URLs:")
    print(f"  Game Server: {game_server_url}")
    print(f"  Coordinator: {coordinator_url}")
    print(f"  Scheduler: {scheduler_url}")
    
    # Setup cleanup handlers to terminate game session on exit
    _cleanup_info["game_server_url"] = game_server_url
    _cleanup_info["coordinator_url"] = coordinator_url
    _cleanup_info["session_id"] = session_id
    _cleanup_info["agent_role"] = agent_role
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    try:
        signal.signal(signal.SIGHUP, signal_handler)
    except (AttributeError, ValueError):
        pass  # SIGHUP not available on some systems
    
    # Register atexit handler
    atexit.register(cleanup_game_session)
    
    # Initialize Phase Coordinator client
    print(f"\nConnecting to Phase Coordinator...")
    phase_client = PhaseCoordinatorClient(
        coordinator_url=coordinator_url,
        session_id=session_id,
        agent_id=agent_id,
        expected_agents=2,
        timeout=args.get("sync_timeout", 300),
    )
    
    # Health check
    if not phase_client.health_check():
        print("✗ Phase Coordinator not responding!")
        print("  Please start it with: python -m opentinker.server.phase_coordinator --port 8791")
        sys.exit(1)
    print("✓ Connected to Phase Coordinator")
    
    # Register with coordinator
    phase_client.register()
    print(f"✓ Registered as {agent_id}")
    
    # Connect to scheduler
    print(f"\nConnecting to scheduler at {scheduler_url}")
    scheduler_client = SchedulerClient(
        scheduler_url=scheduler_url,
        api_key=scheduler_api_key,
    )
    
    # Submit job with agent-specific configuration
    experiment_name = f"{args.experiment_name}_{agent_role.lower()}"
    job_config = OmegaConf.to_container(args, resolve=True)
    job_config["experiment_name"] = experiment_name
    
    print(f"\nSubmitting training job for {agent_role}...")
    job_result = scheduler_client.submit_job(
        config=job_config,
        enable_agent_loop=True,
        wandb_key=args.get("wandb_key"),
        num_gpus=args.get("num_gpus"),
    )
    
    job_id = job_result["job_id"]
    server_url = job_result["server_url"]
    
    # Register for cleanup
    lifecycle.register_job(scheduler_client, job_id)
    
    print(f"\n✓ Job {job_id} allocated!")
    print(f"  Server URL: {server_url}")
    print(f"  GPUs: {job_result.get('gpu_ids')}")
    
    # Create multi-agent interaction config
    interaction_config = args.interaction.config
    interaction_config_path = create_multi_agent_interaction_config(
        game_server_url=game_server_url,
        agent_role=agent_role,
        session_id=session_id,
        board_size=interaction_config.get("board_size", 9),
        max_total_steps=interaction_config.get("max_total_steps", 40),
    )
    print(f"  Interaction config: {interaction_config_path}")
    
    # Override interaction config in args for the environment
    args.interaction.config["interaction_config_path"] = interaction_config_path
    args.interaction.config["agent_role"] = agent_role
    args.interaction.config["session_id"] = session_id
    args.interaction.config["game_server_url"] = game_server_url
    
    # Setup GameEnvironment
    game_kwargs = {
        "board_size": interaction_config.get("board_size", 9),
        "max_total_steps": interaction_config.get("max_total_steps", 40),
    }
    
    print("\nSetting up GameEnvironment...")
    env = GameEnvironment(
        game_class=GomokuGame,
        config=args,
        game_kwargs=game_kwargs,
        job_id=job_id,
    )
    # Setup GameStatsClient - skip for multi-agent mode
    # MultiAgentGameServer doesn't have stats endpoints
    game_stats = None
    print("ℹ Game stats disabled for multi-agent mode")
    
    # Connect to training server
    print(f"\nConnecting to training server at {server_url}")
    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=experiment_name,
        logger_backends=args.logger_backends,
    )
    
    # Set configuration
    client.set_config(args, env)
    
    # Training parameters
    num_steps = args.get("num_steps", None)
    num_epochs = args.get("num_epochs", None)
    
    print("\n" + "=" * 60)
    print(f"Starting training as {agent_role}...")
    if num_steps:
        print(f"  Total steps: {num_steps}")
    elif num_epochs:
        print(f"  Total epochs: {num_epochs}")
    print(f"  Checkpoint frequency: {args.save_freq}")
    print(f"  Validation frequency: {args.test_freq}")
    print("=" * 60)
    
    # Wait for both agents to be ready
    print("\nWaiting for all agents to be ready...")
    phase_client.sync_barrier("initialization")
    print("✓ All agents ready! Starting training loop.")
    
    try:
        # Training loop with phase synchronization
        final_metrics = client.fit(
            env=env,
            num_epochs=num_epochs,
            num_steps=num_steps,
            save_freq=args.save_freq,
            test_freq=args.test_freq,
            verbose=True,
            validate_before_training=True,
            game_stats_client=game_stats,
            phase_client=phase_client,  # Enable step-level sync
        )
        
        print("\n" + "=" * 60)
        print(f"Training completed for {agent_role}!")
        print(f"Final metrics: {final_metrics}")
        
        if game_stats:
            cumulative = game_stats.get_all_stats()
            if cumulative:
                print(f"\nGame Statistics:")
                print(f"  Total games: {cumulative.get('total_games', 0):.0f}")
                print(f"  Win rate: {cumulative.get('cumulative_win_rate', 0):.1%}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print(f"\n⚠ Training interrupted by user ({agent_role})")
    except Exception as e:
        print(f"\n✗ Training failed for {agent_role}: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup game session on game server
        cleanup_game_session()
        
        # Cleanup environment
        env.cleanup()
        try:
            os.unlink(interaction_config_path)
        except:
            pass


if __name__ == "__main__":
    main()
