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
"""Phase Coordinator Client for Multi-Agent Synchronization.

This module provides a client wrapper for interacting with the PhaseCoordinator
service, simplifying barrier synchronization in multi-agent training.

Usage:
    from opentinker.client.utils.phase_coordinator_client import PhaseCoordinatorClient
    
    client = PhaseCoordinatorClient(
        coordinator_url="http://localhost:8791",
        session_id="game_001",
        agent_id="agent_black",
        expected_agents=2,
    )
    
    # Register with coordinator
    client.register()
    
    # Synchronize phases
    client.sync_barrier("rollout")  # Enter and wait for all
    # ... do rollout ...
    client.complete_and_wait("rollout")  # Complete and wait for all
    
    client.sync_barrier("training")
    # ... do training ...
    client.complete_and_wait("training")
"""

import logging
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class PhaseCoordinatorClient:
    """Client for interacting with the Phase Coordinator service."""
    
    def __init__(
        self,
        coordinator_url: str,
        session_id: str,
        agent_id: str,
        expected_agents: int = 2,
        timeout: float = 300,
        retry_delay: float = 1.0,
        max_retries: int = 3,
    ):
        """Initialize the Phase Coordinator client.
        
        Args:
            coordinator_url: URL of the Phase Coordinator service
            session_id: Unique session identifier for this training run
            agent_id: Unique identifier for this agent
            expected_agents: Number of agents expected to join
            timeout: Timeout for barrier operations in seconds
            retry_delay: Delay between retries for transient errors
            max_retries: Maximum number of retries for transient errors
        """
        self.coordinator_url = coordinator_url.rstrip("/")
        self.session_id = session_id
        self.agent_id = agent_id
        self.expected_agents = expected_agents
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        
        self.session = requests.Session()
        self._registered = False
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with retries."""
        url = f"{self.coordinator_url}/{endpoint}"
        timeout = timeout or self.timeout
        
        for attempt in range(self.max_retries):
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, params=params, timeout=timeout)
                elif method.upper() == "POST":
                    response = self.session.post(url, json=json_data, timeout=timeout)
                elif method.upper() == "DELETE":
                    response = self.session.delete(url, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Request to {endpoint} timed out. "
                        f"Retry {attempt + 1}/{self.max_retries}..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise
            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Connection to coordinator failed. "
                        f"Retry {attempt + 1}/{self.max_retries}..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise
    
    def health_check(self) -> bool:
        """Check if the coordinator is healthy."""
        try:
            result = self._make_request("GET", "health", timeout=5.0)
            return result.get("status") == "healthy"
        except Exception as e:
            logger.warning(f"Coordinator health check failed: {e}")
            return False
    
    def register(self) -> Dict[str, Any]:
        """Register this agent with the coordinator.
        
        Returns:
            Registration result including count of registered agents.
        """
        result = self._make_request("POST", "register", json_data={
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "expected_agents": self.expected_agents,
        })
        
        self._registered = True
        logger.info(
            f"[{self.agent_id}] Registered with coordinator. "
            f"({result['registered_count']}/{result['expected_count']})"
        )
        return result
    
    def enter_phase(self, phase: str, step: Optional[int] = None) -> Dict[str, Any]:
        """Signal that this agent is entering a phase.
        
        Args:
            phase: Phase name (e.g., "rollout", "training")
            step: Optional step number for tracking
            
        Returns:
            Result including count of agents that have entered.
        """
        result = self._make_request("POST", "enter_phase", json_data={
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "phase": phase,
            "step": step,
        })
        
        logger.debug(
            f"[{self.agent_id}] Entered {phase}. "
            f"({result['agents_entered']}/{result['expected_agents']})"
        )
        return result
    
    def wait_all_enter(self, phase: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait until all agents have entered the phase.
        
        Args:
            phase: Phase name to wait for
            timeout: Timeout in seconds (uses default if not specified)
            
        Returns:
            Result with list of agents that have entered.
        """
        timeout = timeout or self.timeout
        
        result = self._make_request(
            "GET",
            "wait_all_enter",
            params={
                "session_id": self.session_id,
                "phase": phase,
                "timeout": timeout,
            },
            timeout=timeout + 5,  # Add buffer for network latency
        )
        
        logger.info(
            f"[{self.agent_id}] All agents entered {phase}: {result['agents']}"
        )
        return result
    
    def complete_phase(self, phase: str) -> Dict[str, Any]:
        """Signal that this agent has completed a phase.
        
        Args:
            phase: Phase name that was completed
            
        Returns:
            Result including count of agents that have completed.
        """
        result = self._make_request("POST", "complete_phase", json_data={
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "phase": phase,
        })
        
        logger.debug(
            f"[{self.agent_id}] Completed {phase}. "
            f"({result['agents_completed']}/{result['expected_agents']})"
        )
        return result
    
    def wait_all_complete(self, phase: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait until all agents have completed the phase.
        
        Args:
            phase: Phase name to wait for completion
            timeout: Timeout in seconds (uses default if not specified)
            
        Returns:
            Result with list of agents that have completed.
        """
        timeout = timeout or self.timeout
        
        result = self._make_request(
            "GET",
            "wait_all_complete",
            params={
                "session_id": self.session_id,
                "phase": phase,
                "timeout": timeout,
            },
            timeout=timeout + 5,  # Add buffer for network latency
        )
        
        logger.info(
            f"[{self.agent_id}] All agents completed {phase}: {result['agents']}"
        )
        return result
    
    def sync_barrier(self, phase: str, step: Optional[int] = None, timeout: Optional[float] = None) -> None:
        """Convenience method: enter phase and wait for all agents.
        
        This is the most common pattern for synchronization.
        
        Args:
            phase: Phase name
            step: Optional step number
            timeout: Timeout for waiting
        """
        self.enter_phase(phase, step=step)
        self.wait_all_enter(phase, timeout=timeout)
    
    def complete_and_wait(self, phase: str, timeout: Optional[float] = None) -> None:
        """Convenience method: complete phase and wait for all agents.
        
        Args:
            phase: Phase name
            timeout: Timeout for waiting
        """
        self.complete_phase(phase)
        self.wait_all_complete(phase, timeout=timeout)
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status.
        
        Returns:
            Session status including phase, step, and agent counts.
        """
        return self._make_request(
            "GET",
            f"session/{self.session_id}/status",
        )
    
    def reset_session(self) -> Dict[str, Any]:
        """Reset the session for a new training run.
        
        Returns:
            Reset result.
        """
        return self._make_request(
            "POST",
            f"session/{self.session_id}/reset",
        )
