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
"""Tests for Phase Coordinator.

Run with: pytest opentinker/tests/test_phase_coordinator.py -v
"""

import asyncio
import pytest
import threading
import time
from typing import Set

from opentinker.server.phase_coordinator import (
    Phase,
    PhaseCoordinator,
    SessionState,
)


class TestSessionState:
    """Test SessionState data class."""
    
    def test_session_creation(self):
        """Test creating a new session state."""
        state = SessionState(session_id="test_001", expected_agents=2)
        
        assert state.session_id == "test_001"
        assert state.expected_agents == 2
        assert state.current_phase == Phase.IDLE
        assert len(state.registered_agents) == 0
        assert len(state.agents_entered) == 0
        assert len(state.agents_completed) == 0
    
    def test_reset_phase(self):
        """Test resetting phase tracking."""
        state = SessionState(session_id="test_002", expected_agents=2)
        state.agents_entered.add("agent1")
        state.agents_completed.add("agent1")
        
        state.reset_phase()
        
        assert len(state.agents_entered) == 0
        assert len(state.agents_completed) == 0


class TestPhaseBarrierLogic:
    """Test barrier synchronization logic."""
    
    def test_all_entered_detection(self):
        """Test detection when all agents have entered."""
        state = SessionState(session_id="test_barrier_001", expected_agents=2)
        state.current_phase = Phase.ROLLOUT
        
        # First agent enters
        state.agents_entered.add("agent1")
        assert len(state.agents_entered) < state.expected_agents
        
        # Second agent enters
        state.agents_entered.add("agent2")
        assert len(state.agents_entered) >= state.expected_agents
    
    def test_all_completed_detection(self):
        """Test detection when all agents have completed."""
        state = SessionState(session_id="test_barrier_002", expected_agents=2)
        state.current_phase = Phase.ROLLOUT
        state.agents_entered = {"agent1", "agent2"}
        
        # First agent completes
        state.agents_completed.add("agent1")
        assert len(state.agents_completed) < state.expected_agents
        
        # Second agent completes
        state.agents_completed.add("agent2")
        assert len(state.agents_completed) >= state.expected_agents
    
    def test_phase_transition(self):
        """Test phase transition resets tracking."""
        state = SessionState(session_id="test_barrier_003", expected_agents=2)
        
        # Complete rollout phase
        state.current_phase = Phase.ROLLOUT
        state.agents_entered = {"agent1", "agent2"}
        state.agents_completed = {"agent1", "agent2"}
        
        # Transition to training - should reset
        state.reset_phase()
        state.current_phase = Phase.TRAINING
        
        assert len(state.agents_entered) == 0
        assert len(state.agents_completed) == 0
        assert state.current_phase == Phase.TRAINING


class TestCoordinatorBasics:
    """Test basic coordinator functionality."""
    
    def test_coordinator_creation(self):
        """Test creating a coordinator."""
        coordinator = PhaseCoordinator()
        
        assert coordinator.app is not None
        assert len(coordinator.sessions) == 0
    
    def test_session_storage(self):
        """Test session storage in coordinator."""
        coordinator = PhaseCoordinator()
        
        state = SessionState(session_id="test_storage_001", expected_agents=2)
        coordinator.sessions["test_storage_001"] = state
        
        assert "test_storage_001" in coordinator.sessions
        assert coordinator.sessions["test_storage_001"].expected_agents == 2


class TestBarrierSynchronization:
    """Test barrier synchronization with multiple threads."""
    
    def test_barrier_blocks_until_all_enter(self):
        """Test that barrier blocks until all agents enter."""
        state = SessionState(session_id="test_sync_001", expected_agents=2)
        state.current_phase = Phase.ROLLOUT
        
        results = []
        
        def agent_task(agent_id: str, delay: float):
            """Simulate an agent entering the phase."""
            time.sleep(delay)
            state.agents_entered.add(agent_id)
            if len(state.agents_entered) >= state.expected_agents:
                state.all_entered_event.set()
            results.append(f"{agent_id}_entered")
        
        def waiter_task(agent_id: str):
            """Simulate waiting for all agents."""
            # Use loop for checking since regular events don't work across threads
            start = time.time()
            while len(state.agents_entered) < state.expected_agents:
                if time.time() - start > 5:
                    raise TimeoutError()
                time.sleep(0.01)
            results.append(f"{agent_id}_passed_barrier")
        
        # Start threads
        t1 = threading.Thread(target=agent_task, args=("agent1", 0.1))
        t2 = threading.Thread(target=agent_task, args=("agent2", 0.3))
        t3 = threading.Thread(target=waiter_task, args=("agent1",))
        t4 = threading.Thread(target=waiter_task, args=("agent2",))
        
        t1.start()
        t3.start()
        t2.start()
        t4.start()
        
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        
        # Both agents should have entered and passed barrier
        assert "agent1_entered" in results
        assert "agent2_entered" in results
        assert "agent1_passed_barrier" in results
        assert "agent2_passed_barrier" in results
    
    def test_completion_barrier(self):
        """Test completion barrier."""
        state = SessionState(session_id="test_sync_002", expected_agents=2)
        state.current_phase = Phase.ROLLOUT
        state.agents_entered = {"agent1", "agent2"}
        
        results = []
        
        def complete_task(agent_id: str, delay: float):
            """Simulate completing a phase."""
            time.sleep(delay)
            state.agents_completed.add(agent_id)
            if len(state.agents_completed) >= state.expected_agents:
                state.all_completed_event.set()
            results.append(f"{agent_id}_completed")
        
        def waiter_task(agent_id: str):
            """Wait for all to complete."""
            start = time.time()
            while len(state.agents_completed) < state.expected_agents:
                if time.time() - start > 5:
                    raise TimeoutError()
                time.sleep(0.01)
            results.append(f"{agent_id}_all_done")
        
        t1 = threading.Thread(target=complete_task, args=("agent1", 0.1))
        t2 = threading.Thread(target=complete_task, args=("agent2", 0.2))
        t3 = threading.Thread(target=waiter_task, args=("agent1",))
        t4 = threading.Thread(target=waiter_task, args=("agent2",))
        
        t1.start()
        t3.start()
        t2.start()
        t4.start()
        
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        
        assert "agent1_completed" in results
        assert "agent2_completed" in results
        assert "agent1_all_done" in results
        assert "agent2_all_done" in results


class TestPhaseFlow:
    """Test complete phase flow."""
    
    def test_rollout_to_training_transition(self):
        """Test transitioning from rollout to training phase."""
        state = SessionState(session_id="test_flow_001", expected_agents=2)
        
        # Rollout phase
        state.current_phase = Phase.ROLLOUT
        state.agents_entered = {"agent1", "agent2"}
        state.agents_completed = {"agent1", "agent2"}
        
        # Verify rollout complete
        assert len(state.agents_completed) >= state.expected_agents
        
        # Transition to training
        state.reset_phase()
        state.current_phase = Phase.TRAINING
        
        # Verify reset
        assert len(state.agents_entered) == 0
        assert len(state.agents_completed) == 0
        assert state.current_phase == Phase.TRAINING
        
        # Training phase
        state.agents_entered = {"agent1", "agent2"}
        state.agents_completed = {"agent1", "agent2"}
        
        # Verify training complete
        assert len(state.agents_completed) >= state.expected_agents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
