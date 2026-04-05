import asyncio
import logging
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
connected_clients: set[WebSocket] = set()
logger = logging.getLogger("oceanpulse.websocket")


async def _heartbeat(ws: WebSocket):
    while True:
        try:
            await ws.send_json({"type": "ping"})
        except WebSocketDisconnect:
            break
        except Exception:
            logger.info("websocket_heartbeat_stopped")
            break
        await asyncio.sleep(30)


@router.websocket("/ws/live")
async def live_feed(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    client = ws.client.host if ws.client else "unknown"
    logger.info("websocket_connected client=%s active_clients=%s", client, len(connected_clients))
    heartbeat_task = asyncio.create_task(_heartbeat(ws))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        logger.info("websocket_disconnected client=%s", client)
    finally:
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError, WebSocketDisconnect):
            await heartbeat_task
        connected_clients.discard(ws)
        logger.info("websocket_cleanup client=%s active_clients=%s", client, len(connected_clients))


async def broadcast(message: dict):
    dead: list[WebSocket] = []
    for client in connected_clients:
        try:
            await client.send_json(message)
        except Exception:
            logger.exception("websocket_broadcast_failed")
            dead.append(client)
    for client in dead:
        connected_clients.discard(client)
