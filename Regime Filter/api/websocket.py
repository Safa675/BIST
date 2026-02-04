"""
WebSocket manager for real-time regime updates
"""

from fastapi import WebSocket
from typing import List, Dict, Any
import asyncio
import json
from datetime import datetime


class ConnectionManager:
    """
    Manages WebSocket connections and broadcasts

    Features:
    - Connection tracking
    - Broadcast to all clients
    - Heartbeat to keep connections alive
    - JSON message formatting
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept and track new connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = {
            'client_id': client_id or f"client_{len(self.active_connections)}",
            'connected_at': datetime.now(),
            'last_message': datetime.now()
        }
        print(f"WebSocket connected: {self.connection_info[websocket]['client_id']}")

    def disconnect(self, websocket: WebSocket):
        """Remove disconnected client"""
        if websocket in self.active_connections:
            client_id = self.connection_info.get(websocket, {}).get('client_id', 'unknown')
            self.active_connections.remove(websocket)
            if websocket in self.connection_info:
                del self.connection_info[websocket]
            print(f"WebSocket disconnected: {client_id}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
            if websocket in self.connection_info:
                self.connection_info[websocket]['last_message'] = datetime.now()
        except Exception as e:
            print(f"Error sending message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        disconnected = []

        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
                if websocket in self.connection_info:
                    self.connection_info[websocket]['last_message'] = datetime.now()
            except Exception as e:
                print(f"Broadcast error: {e}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws)

    async def broadcast_regime_update(self, regime_data: Dict[str, Any]):
        """
        Broadcast regime update to all clients

        Args:
            regime_data: Dict with regime prediction info
        """
        message = {
            'type': 'regime_update',
            'timestamp': datetime.now().isoformat(),
            'data': regime_data
        }
        await self.broadcast(message)

    async def broadcast_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """
        Broadcast alert to all clients

        Args:
            alert_type: Type of alert (regime_change, stress_warning, etc.)
            alert_data: Alert details
        """
        message = {
            'type': 'alert',
            'alert_type': alert_type,
            'timestamp': datetime.now().isoformat(),
            'data': alert_data
        }
        await self.broadcast(message)

    async def send_heartbeat(self):
        """Send heartbeat to all connections"""
        message = {
            'type': 'heartbeat',
            'timestamp': datetime.now().isoformat(),
            'data': {'active_connections': len(self.active_connections)}
        }
        await self.broadcast(message)

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)

    def get_connection_info(self) -> List[Dict]:
        """Get info about all connections"""
        return [
            {
                'client_id': info['client_id'],
                'connected_at': info['connected_at'].isoformat(),
                'last_message': info['last_message'].isoformat()
            }
            for info in self.connection_info.values()
        ]


# Global connection manager instance
manager = ConnectionManager()


async def heartbeat_task(interval: int = 30):
    """
    Background task to send periodic heartbeats

    Args:
        interval: Seconds between heartbeats
    """
    while True:
        await asyncio.sleep(interval)
        if manager.get_connection_count() > 0:
            await manager.send_heartbeat()
