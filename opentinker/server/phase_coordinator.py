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
"""Phase Coordinator for Multi-Agent Training Synchronization.

This module provides a barrier synchronization service for coordinating
rollout and training phases across multiple agents.

Usage:
    # Start the coordinator server
    python -m opentinker.server.phase_coordinator --port 8791
    
    # Or import and use programmatically
    from opentinker.server.phase_coordinator import PhaseCoordinator
    coordinator = PhaseCoordinator()
    coordinator.run(port=8791)
"""

import asyncio
import logging
import os
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class Phase(str, Enum):
    """Training phases."""
    IDLE = "idle"
    INITIALIZATION = "initialization"
    ROLLOUT = "rollout"
    TRAINING = "training"


@dataclass
class SessionState:
    """State for a coordination session."""
    session_id: str
    expected_agents: int = 2
    
    # Registered agents
    registered_agents: Set[str] = field(default_factory=set)
    
    # Phase tracking
    current_phase: Phase = Phase.IDLE
    agents_entered: Set[str] = field(default_factory=set)
    agents_completed: Set[str] = field(default_factory=set)
    
    # Step tracking
    current_step: int = 0
    
    # Synchronization events
    all_entered_event: asyncio.Event = field(default_factory=asyncio.Event)
    all_completed_event: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    def reset_phase(self):
        """Reset phase tracking for next phase."""
        self.agents_entered.clear()
        self.agents_completed.clear()
        self.all_entered_event.clear()
        self.all_completed_event.clear()


# ==================== Pydantic Models ====================

class RegisterAgentRequest(BaseModel):
    """Request to register an agent."""
    session_id: str
    agent_id: str
    expected_agents: int = 2


class RegisterAgentResponse(BaseModel):
    """Response after registering."""
    session_id: str
    agent_id: str
    registered_count: int
    expected_count: int
    status: str


class EnterPhaseRequest(BaseModel):
    """Request to enter a phase."""
    session_id: str
    agent_id: str
    phase: str
    step: Optional[int] = None


class EnterPhaseResponse(BaseModel):
    """Response after entering a phase."""
    session_id: str
    phase: str
    agents_entered: int
    expected_agents: int
    all_entered: bool


class WaitAllEnterResponse(BaseModel):
    """Response after all agents entered."""
    session_id: str
    phase: str
    agents: list
    status: str


class CompletePhaseRequest(BaseModel):
    """Request to complete a phase."""
    session_id: str
    agent_id: str
    phase: str


class CompletePhaseResponse(BaseModel):
    """Response after completing a phase."""
    session_id: str
    phase: str
    agents_completed: int
    expected_agents: int
    all_completed: bool


class WaitAllCompleteResponse(BaseModel):
    """Response after all agents completed."""
    session_id: str
    phase: str
    agents: list
    status: str


class SessionStatusResponse(BaseModel):
    """Response with session status."""
    session_id: str
    expected_agents: int
    registered_agents: list
    current_phase: str
    current_step: int
    agents_entered: list
    agents_completed: list


# ==================== Phase Coordinator ====================

class PhaseCoordinator:
    """HTTP server for multi-agent phase synchronization."""
    
    def __init__(self):
        self.app = FastAPI(title="Phase Coordinator")
        self.sessions: Dict[str, SessionState] = {}
        self.lock = threading.Lock()
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "sessions": len(self.sessions)}
        
        @self.app.post("/register", response_model=RegisterAgentResponse)
        async def register_agent(request: RegisterAgentRequest):
            """Register an agent with the coordinator."""
            session_id = request.session_id
            
            # Create session if doesn't exist
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionState(
                    session_id=session_id,
                    expected_agents=request.expected_agents,
                )
            
            session = self.sessions[session_id]
            
            async with session.lock:
                session.registered_agents.add(request.agent_id)
                
                logger.info(
                    f"[{session_id}] Agent {request.agent_id} registered. "
                    f"({len(session.registered_agents)}/{session.expected_agents})"
                )
                
                return RegisterAgentResponse(
                    session_id=session_id,
                    agent_id=request.agent_id,
                    registered_count=len(session.registered_agents),
                    expected_count=session.expected_agents,
                    status="registered",
                )
        
        @self.app.post("/enter_phase", response_model=EnterPhaseResponse)
        async def enter_phase(request: EnterPhaseRequest):
            """Signal that an agent is entering a phase."""
            session_id = request.session_id
            
            if session_id not in self.sessions:
                raise HTTPException(404, f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            phase = Phase(request.phase)
            
            async with session.lock:
                # Check if agent is registered
                if request.agent_id not in session.registered_agents:
                    raise HTTPException(400, f"Agent {request.agent_id} not registered")
                
                # Handle phase transition
                if session.current_phase != phase:
                    # New phase - reset tracking
                    session.current_phase = phase
                    session.agents_entered.clear()
                    session.agents_completed.clear()
                    session.all_entered_event.clear()
                    session.all_completed_event.clear()
                    if request.step is not None:
                        session.current_step = request.step
                
                # Mark agent as entered
                session.agents_entered.add(request.agent_id)
                
                # Check if all agents entered
                all_entered = len(session.agents_entered) >= session.expected_agents
                if all_entered:
                    session.all_entered_event.set()
                
                logger.info(
                    f"[{session_id}] Agent {request.agent_id} entered {phase.value}. "
                    f"({len(session.agents_entered)}/{session.expected_agents})"
                )
                
                return EnterPhaseResponse(
                    session_id=session_id,
                    phase=phase.value,
                    agents_entered=len(session.agents_entered),
                    expected_agents=session.expected_agents,
                    all_entered=all_entered,
                )
        
        @self.app.get("/wait_all_enter", response_model=WaitAllEnterResponse)
        async def wait_all_enter(
            session_id: str,
            phase: str,
            timeout: float = 300,
        ):
            """Wait until all agents have entered the phase."""
            if session_id not in self.sessions:
                raise HTTPException(404, f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            target_phase = Phase(phase)
            
            # Fast path: already all entered
            if (
                session.current_phase == target_phase
                and len(session.agents_entered) >= session.expected_agents
            ):
                return WaitAllEnterResponse(
                    session_id=session_id,
                    phase=phase,
                    agents=list(session.agents_entered),
                    status="all_entered",
                )
            
            # Wait for all agents
            try:
                await asyncio.wait_for(
                    session.all_entered_event.wait(),
                    timeout=timeout,
                )
                return WaitAllEnterResponse(
                    session_id=session_id,
                    phase=phase,
                    agents=list(session.agents_entered),
                    status="all_entered",
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    408,
                    f"Timeout waiting for all agents to enter {phase}. "
                    f"Entered: {list(session.agents_entered)}, "
                    f"Expected: {session.expected_agents}",
                )
        
        @self.app.post("/complete_phase", response_model=CompletePhaseResponse)
        async def complete_phase(request: CompletePhaseRequest):
            """Signal that an agent has completed a phase."""
            session_id = request.session_id
            
            if session_id not in self.sessions:
                raise HTTPException(404, f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            phase = Phase(request.phase)
            
            async with session.lock:
                # Validate phase
                if session.current_phase != phase:
                    raise HTTPException(
                        400,
                        f"Cannot complete {phase.value}. Current phase: {session.current_phase.value}",
                    )
                
                # Mark agent as completed
                session.agents_completed.add(request.agent_id)
                
                # Check if all agents completed
                all_completed = len(session.agents_completed) >= session.expected_agents
                if all_completed:
                    session.all_completed_event.set()
                
                logger.info(
                    f"[{session_id}] Agent {request.agent_id} completed {phase.value}. "
                    f"({len(session.agents_completed)}/{session.expected_agents})"
                )
                
                return CompletePhaseResponse(
                    session_id=session_id,
                    phase=phase.value,
                    agents_completed=len(session.agents_completed),
                    expected_agents=session.expected_agents,
                    all_completed=all_completed,
                )
        
        @self.app.get("/wait_all_complete", response_model=WaitAllCompleteResponse)
        async def wait_all_complete(
            session_id: str,
            phase: str,
            timeout: float = 300,
        ):
            """Wait until all agents have completed the phase."""
            if session_id not in self.sessions:
                raise HTTPException(404, f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            
            # Fast path: already all completed
            if len(session.agents_completed) >= session.expected_agents:
                return WaitAllCompleteResponse(
                    session_id=session_id,
                    phase=phase,
                    agents=list(session.agents_completed),
                    status="all_completed",
                )
            
            # Wait for all agents
            try:
                await asyncio.wait_for(
                    session.all_completed_event.wait(),
                    timeout=timeout,
                )
                return WaitAllCompleteResponse(
                    session_id=session_id,
                    phase=phase,
                    agents=list(session.agents_completed),
                    status="all_completed",
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    408,
                    f"Timeout waiting for all agents to complete {phase}. "
                    f"Completed: {list(session.agents_completed)}, "
                    f"Expected: {session.expected_agents}",
                )
        
        @self.app.get("/session/{session_id}/status", response_model=SessionStatusResponse)
        async def get_session_status(session_id: str):
            """Get current session status."""
            if session_id not in self.sessions:
                raise HTTPException(404, f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            return SessionStatusResponse(
                session_id=session_id,
                expected_agents=session.expected_agents,
                registered_agents=list(session.registered_agents),
                current_phase=session.current_phase.value,
                current_step=session.current_step,
                agents_entered=list(session.agents_entered),
                agents_completed=list(session.agents_completed),
            )
        
        @self.app.delete("/session/{session_id}")
        async def delete_session(session_id: str):
            """Delete a coordination session."""
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Deleted session {session_id}")
            return {"status": "deleted", "session_id": session_id}
        
        @self.app.post("/session/{session_id}/reset")
        async def reset_session(session_id: str):
            """Reset a session for a new training run."""
            if session_id not in self.sessions:
                raise HTTPException(404, f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            async with session.lock:
                session.current_phase = Phase.IDLE
                session.current_step = 0
                session.agents_entered.clear()
                session.agents_completed.clear()
                session.all_entered_event.clear()
                session.all_completed_event.clear()
            
            logger.info(f"Reset session {session_id}")
            return {"status": "reset", "session_id": session_id}
    
    def run(self, host: str = "0.0.0.0", port: int = 8791):
        """Run the coordinator server."""
        logger.info(f"Starting Phase Coordinator on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase Coordinator for Multi-Agent Training")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8791, help="Port to bind to")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    coordinator = PhaseCoordinator()
    coordinator.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
