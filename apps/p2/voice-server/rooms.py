"""Authenticated /room WebSocket: one room per conversation, one microphone per connection."""

import asyncio
import json
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import unquote
from uuid import uuid4

import numpy as np
from aiohttp import WSMsgType, web

from engines import Engine

ROOM = re.compile(r"p2-[A-Za-z0-9_-]{3,160}")
ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
LANGUAGE = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8}){0,3}")
MIC_RATE = 16000
MAX_MIC_BYTES = 16000  # Browser preroll is split at this limit.
MAX_CONNECTIONS = 64
MAX_ROOM_CONNECTIONS = 16
MAX_MEMBER_CONNECTIONS = 4
MAX_SPEAK_TEXT = 2000
MAX_SPEAK_JOBS = 16
RECENT_IDS = 256
WRITE_QUEUE = 96
MIC_QUEUE_BYTES = MIC_RATE * 2 * 4  # Four seconds of PCM16, regardless of WebSocket frame sizes.
MIC_QUEUE_ITEMS = 256  # Also bound the overhead of tiny audio frames or microphone commands.
MIC_BATCH_BYTES = MIC_RATE * 2 // 5  # Merge up to 200 ms at a time; allow larger preroll frames on their own.
TTS_FRAME_SAMPLES = 8192
TTS_BUFFER = 8
LOG = logging.getLogger(__name__)


class Recognizer(Protocol):
    def feed(self, pcm: bytes) -> list[dict]: ...
    def reset(self) -> None: ...
    def set_pause(self, seconds: float) -> None: ...


@dataclass(frozen=True)
class Participant:
    id: str
    name: str
    role: str

    def member(self, count: int) -> dict:
        return {"id": self.id, "name": self.name, "role": self.role, "connections": count}


@dataclass(frozen=True)
class Voice:
    engine: Engine
    name: str
    speed: float | None

    def config(self) -> dict:
        return {"engine": self.engine.id, "voice": f"{self.engine.id}-{self.name}", "speed": self.speed}

    def utterance(self, id: str) -> dict:
        return {"id": id, "engine": self.engine.id, "voice": f"{self.engine.id}-{self.name}",
                "sampleRate": self.engine.sample_rate}


@dataclass
class SpeakJob:
    id: str
    text: str
    voice: Voice
    stop: threading.Event = field(default_factory=threading.Event)


@dataclass
class Meter:
    rate: int
    burst: int
    tokens: float = field(init=False)
    updated: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.tokens = self.burst

    def take(self, cost: int) -> bool:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if cost > self.tokens:
            return False
        self.tokens -= cost
        return True


def identity(request: web.Request) -> tuple[str, Participant]:
    """Validate p2's authenticated headers, rejecting ambiguous repeated values."""
    values = [request.headers.getall(key, ()) for key in
              ("X-Room", "X-Participant-Id", "X-Participant-Name", "X-Participant-Role")]
    if any(len(value) != 1 for value in values):
        raise ValueError("expected one of each room and participant header")
    room, id, encoded, role = (value[0] for value in values)
    if not ROOM.fullmatch(room) or not ID.fullmatch(id):
        raise ValueError("invalid room or participant id")
    if role not in ("owner", "guest") or (id == "owner" and role != "owner"):
        raise ValueError("invalid participant role")
    if len(encoded) > 64 * 12 or not encoded.isascii() or re.search(r"%(?![0-9A-Fa-f]{2})", encoded):
        raise ValueError("invalid participant name")
    name = unquote(encoded, errors="strict")
    if not 1 <= len(name) <= 64 or not name.strip() or re.search(r"[\x00-\x1f\x7f-\x9f]", name):
        raise ValueError("invalid participant name")
    return room, Participant(id, name, role)


def audio_prefix(kind: int, id: str) -> bytes:
    return bytes((kind, len(id))) + id.encode("ascii") + b"\0" * (len(id) % 2)


class Rooms:
    """Serve the room endpoint behind the speech service's Bearer-token middleware."""

    def __init__(self, recognize: Callable[[], Recognizer], resolve: Callable[[object, object], tuple[Engine, str]],
                 default: dict, tts_pool: ThreadPoolExecutor):
        self.recognize = recognize
        self.resolve = resolve
        engine, voice = resolve(default["engine"], default["voice"])
        self.default = Voice(engine, voice, default["speed"])
        self.tts_pool = tts_pool
        # Microphones share one native recognizer, which already uses multiple inference threads.
        self.stt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="room-stt")
        self.stt_slots = asyncio.Semaphore(1)
        # Leave capacity for the existing synthesis endpoints. Only one room at a time takes an engine's model lock.
        self.tts_slots = asyncio.Semaphore(2)
        self.engine_gates: dict[str, asyncio.Lock] = {}
        self.active: dict[str, Room] = {}
        self.connections = 0
        self.closing = False

    async def room(self, request: web.Request) -> web.WebSocketResponse:
        try:
            name, participant = identity(request)
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from None
        self.check_admission(name, participant)
        socket = web.WebSocketResponse(heartbeat=20, timeout=2, compress=False, max_msg_size=64 * 1024)
        await socket.prepare(request)
        try:
            self.check_admission(name, participant)
        except web.HTTPException:
            await socket.close(code=1013, message=b"Room unavailable", drain=False)
            return socket
        room = self.active.get(name)
        if room is None:
            room = self.active[name] = Room(self, name)
        connection = Connection(room, participant, socket)
        self.connections += 1
        room.join(connection)
        try:
            await connection.run()
        finally:
            room.leave(connection)
            self.connections -= 1
            if not room.connections:
                del self.active[name]
                await room.close()
            code, message = connection.stopped.result() if connection.stopped.done() else (1000, b"")
            await socket.close(code=code, message=message, drain=False)
        return socket

    def check_admission(self, name: str, participant: Participant) -> None:
        if self.closing or self.connections >= MAX_CONNECTIONS:
            raise web.HTTPServiceUnavailable(text="voice rooms are full")
        room = self.active.get(name)
        if room is None:
            return
        same = room.members(participant.id)
        if same and same[0].participant.role != participant.role:
            raise web.HTTPBadRequest(text="conflicting participant role")
        if len(room.connections) >= MAX_ROOM_CONNECTIONS or len(same) >= MAX_MEMBER_CONNECTIONS:
            raise web.HTTPServiceUnavailable(text="voice room is full")

    async def shutdown(self, _app: web.Application) -> None:
        """Register with aiohttp's on_shutdown to signal rooms before waiting for requests to finish."""
        self.closing = True
        for room in self.active.values():
            for peer in room.connections:
                peer.stop(1001, b"Server shutting down")
        self.stt_pool.shutdown(wait=False, cancel_futures=True)

    async def recognize_work(self, work, *args):
        await self.stt_slots.acquire()
        if self.closing:
            self.stt_slots.release()
            raise asyncio.CancelledError
        future = asyncio.get_running_loop().run_in_executor(self.stt_pool, work, *args)

        def finished(result: asyncio.Future) -> None:
            self.stt_slots.release()
            if not result.cancelled():
                result.exception()  # A disconnected microphone may no longer be waiting for this result.

        future.add_done_callback(finished)
        return await asyncio.shield(future)

    async def stream(self, room: "Room", job: SpeakJob) -> None:
        engine = job.voice.engine
        gate = self.engine_gates.get(engine.id)
        if gate is None:
            gate = self.engine_gates[engine.id] = asyncio.Lock()
        await gate.acquire()
        try:
            await self.tts_slots.acquire()
        except BaseException:
            gate.release()
            raise
        future = None
        try:
            loop = asyncio.get_running_loop()
            chunks: asyncio.Queue[bytes | None] = asyncio.Queue(TTS_BUFFER + 1)
            # Reserve the extra queue entry for the end marker. The semaphore also bounds queued loop callbacks.
            credits = threading.BoundedSemaphore(TTS_BUFFER)
            prefix = audio_prefix(1, job.id)

            def produce() -> None:
                try:
                    if job.stop.is_set():
                        return
                    for chunk in engine.stream(job.text, job.voice.name, job.voice.speed, job.stop):
                        for start in range(0, len(chunk), TTS_FRAME_SAMPLES):
                            while not job.stop.is_set():
                                if credits.acquire(timeout=0.05):
                                    break
                            if job.stop.is_set():
                                return
                            samples = chunk[start:start + TTS_FRAME_SAMPLES]
                            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                            loop.call_soon_threadsafe(chunks.put_nowait, prefix + pcm)
                finally:
                    if not job.stop.is_set():
                        loop.call_soon_threadsafe(chunks.put_nowait, None)

            room.current = job.voice.utterance(job.id)
            room.broadcast({"type": "tts_start", **room.current})
            future = loop.run_in_executor(self.tts_pool, produce)
            while (frame := await chunks.get()) is not None:
                credits.release()
                room.broadcast(frame)
            await asyncio.shield(future)
        finally:
            def finished(result: asyncio.Future | None) -> None:
                self.tts_slots.release()
                gate.release()
                if result is not None and not result.cancelled():
                    result.exception()  # Cancellation may outlive inference; never release its slots early.

            if future is None:
                finished(None)
            else:
                future.add_done_callback(finished)


class Room:
    """The room owns membership and speech; request handlers own their microphones."""

    def __init__(self, service: Rooms, name: str):
        self.service = service
        self.name = name
        self.voice = service.default
        self.connections: dict[Connection, None] = {}
        self.current: dict | None = None
        self.jobs: asyncio.Queue[SpeakJob] = asyncio.Queue(MAX_SPEAK_JOBS)
        self.pending: dict[str, SpeakJob] = {}
        self.recent: dict[str, None] = {}
        self.synthesis: asyncio.Task | None = None
        self.running: SpeakJob | None = None
        self.closed = False
        self.runner = asyncio.create_task(self.speak_in_order())

    def members(self, id: str) -> list["Connection"]:
        return [peer for peer in self.connections if peer.participant.id == id]

    def join(self, peer: "Connection") -> None:
        self.connections[peer] = None
        groups: dict[str, list[Connection]] = {}
        for connection in self.connections:
            groups.setdefault(connection.participant.id, []).append(connection)
        count = len(groups[peer.participant.id])
        member = peer.participant.member(count)
        peer.send({"type": "welcome", "room": self.name, "you": {**member, "connectionId": uuid4().hex},
                   "members": [group[-1].participant.member(len(group)) for group in groups.values()],
                   "tts": self.voice.config(), "current": self.current, "micSampleRate": MIC_RATE})
        changed = "member_joined" if count == 1 else "member_updated"
        for connection in self.connections:
            if connection is not peer:
                connection.send({"type": changed, "member": member})
        for id, group in groups.items():
            if any(connection.talking for connection in group):
                peer.send({"type": "talking", "participant": id, "on": True})

    def leave(self, peer: "Connection") -> None:
        del self.connections[peer]
        self.talking(peer, False)
        rest = self.members(peer.participant.id)
        member = rest[-1].participant if rest else peer.participant
        changed = "member_updated" if rest else "member_left"
        self.broadcast({"type": changed, "member": member.member(len(rest))})

    async def close(self) -> None:
        self.closed = True
        self.cancel(None)
        self.runner.cancel()
        await asyncio.gather(self.runner, return_exceptions=True)

    def broadcast(self, frame: dict | bytes) -> None:
        for peer in self.connections:
            peer.send(frame)

    def relay(self, id: str, frame: bytes) -> None:
        framed = audio_prefix(2, id) + frame
        for peer in self.connections:
            if peer.participant.id != id:
                peer.send(framed)

    def talking(self, peer: "Connection", on: bool) -> None:
        if peer.talking == on:
            return
        peer.talking = on
        # Several tabs can speak independently; the room indicator belongs to their shared principal.
        id = peer.participant.id
        if not any(other is not peer and other.talking for other in self.members(id)):
            self.broadcast({"type": "talking", "participant": id, "on": on})

    def config(self, body: dict) -> None:
        for key in ("engine", "voice"):
            if key in body and not valid_id(body[key]):
                raise ValueError(f"invalid {key}")
        if "language" in body and (not isinstance(body["language"], str) or not LANGUAGE.fullmatch(body["language"])):
            raise ValueError("invalid language")
        # The existing engines are English-only; they don't advertise or accept a language selector.
        if "engine" in body or "voice" in body:
            engine, name = self.service.resolve(body.get("engine"), body.get("voice"))
        else:
            engine, name = self.voice.engine, self.voice.name
        speed = bounded_number(body["speed"], 0.5, 2.0, "speed") if "speed" in body else self.voice.speed
        self.voice = Voice(engine, name, speed)
        self.broadcast({"type": "config", **self.voice.config()})

    def speak(self, peer: "Connection", id: str, text: str) -> None:
        duplicate = id in self.pending or id in self.recent
        if not duplicate:
            if len(self.pending) >= MAX_SPEAK_JOBS:
                peer.error("tts_busy", "Too many utterances queued", id)
                return
            job = SpeakJob(id, text, self.voice)
            self.pending[id] = job
            self.jobs.put_nowait(job)
        peer.send({"type": "speak_ack", "id": id, "duplicate": duplicate})

    def cancel(self, id: str | None) -> None:
        for job in self.pending.values():
            if id is None or job.id == id:
                job.stop.set()
                if job is self.running and self.synthesis is not None:
                    self.synthesis.cancel()

    async def speak_in_order(self) -> None:
        while True:
            job = await self.jobs.get()
            failure = False
            if not job.stop.is_set():
                self.running = job
                self.synthesis = asyncio.create_task(self.service.stream(self, job))
                try:
                    await self.synthesis
                except asyncio.CancelledError:
                    if self.closed or not job.stop.is_set():
                        raise
                except Exception:
                    LOG.exception("Room synthesis failed")
                    failure = not job.stop.is_set()
                finally:
                    self.current = self.synthesis = self.running = None
            del self.pending[job.id]
            self.recent[job.id] = None
            if len(self.recent) > RECENT_IDS:
                del self.recent[next(iter(self.recent))]
            if failure:
                self.broadcast({"type": "tts_error", "id": job.id, "code": "synthesis_failed",
                                "message": "Could not synthesize the utterance"})
            else:
                self.broadcast({"type": "tts_done", "id": job.id, "cancelled": job.stop.is_set()})


def valid_id(value: object) -> bool:
    return isinstance(value, str) and ID.fullmatch(value) is not None


def bounded_number(value: object, low: float, high: float, label: str) -> float:
    if type(value) not in (float, int) or not low <= value <= high:
        raise ValueError(f"invalid {label}")
    return float(value)


class Connection:
    """One ordered outbox and one ordered stream of PCM, reset and pause commands per socket."""

    def __init__(self, room: Room, participant: Participant, socket: web.WebSocketResponse):
        self.room = room
        self.participant = participant
        self.socket = socket
        self.outbox: asyncio.Queue[dict | bytes] = asyncio.Queue(WRITE_QUEUE)
        # Raw PCM bytes, None (reset), or float (new pause). All are handled in socket arrival order.
        self.mic: deque[bytes | None | float] = deque()
        self.mic_bytes = 0
        self.mic_ready = asyncio.Event()
        self.stopped: asyncio.Future[tuple[int, bytes]] = asyncio.get_running_loop().create_future()
        self.audio_meter = Meter(rate=48000, burst=64000)
        self.command_meter = Meter(rate=32, burst=64)
        self.last_rate_error = -float("inf")
        self.talking = False

    async def run(self) -> None:
        tasks = [asyncio.create_task(coro) for coro in (self.read(), self.write(), self.recognize())]
        try:
            await asyncio.wait([*tasks, self.stopped], return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            for result in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(result, Exception) and not isinstance(result, ConnectionError):
                    LOG.error("Voice room connection failed", exc_info=result)
                    self.stop(1011, b"Voice room failed")

    def stop(self, code: int, message: bytes) -> None:
        if not self.stopped.done():
            self.stopped.set_result((code, message))

    def send(self, message: dict | bytes) -> None:
        if self.stopped.done():
            return
        try:
            self.outbox.put_nowait(message)
        except asyncio.QueueFull:
            self.stop(1013, b"Slow room connection")

    def error(self, code: str, message: str, id: str | None = None) -> None:
        error = {"type": "error", "code": code, "message": message}
        if id is not None:
            error["id"] = id
        self.send(error)

    async def write(self) -> None:
        while True:
            message = await self.outbox.get()
            if isinstance(message, bytes):
                await self.socket.send_bytes(message)
            else:
                await self.socket.send_json(message)

    async def read(self) -> None:
        buffered = 0
        async for message in self.socket:
            buffered += 1
            if buffered == 16:
                # A buffered sender must not fill other peers' outboxes before their writers run.
                await asyncio.sleep(0)
                buffered = 0
            if self.stopped.done():
                return
            if message.type == WSMsgType.BINARY:
                pcm = message.data
                if not self.audio_meter.take(max(len(pcm), 320)):
                    now = time.monotonic()
                    if now - self.last_rate_error >= 1:
                        self.last_rate_error = now
                        self.error("mic_rate", "Microphone audio is arriving too fast")
                elif not pcm or len(pcm) > MAX_MIC_BYTES or len(pcm) % 2:
                    self.error("invalid_message", "Expected at most 16000 bytes of 16 kHz PCM16")
                elif self.queue_mic(pcm):
                    self.room.relay(self.participant.id, pcm)
            elif message.type == WSMsgType.TEXT:
                if not self.command_meter.take(1):
                    self.stop(1008, b"Too many room commands")
                    return
                self.command(message.data)
            else:
                return

    def queue_mic(self, message: bytes | None | float) -> bool:
        size = len(message) if isinstance(message, bytes) else 0
        if len(self.mic) >= MIC_QUEUE_ITEMS or self.mic_bytes + size > MIC_QUEUE_BYTES:
            self.stop(1013, b"Microphone processing is overloaded")
            return False
        self.mic.append(message)
        self.mic_bytes += size
        self.mic_ready.set()
        return True

    def command(self, text: str) -> None:
        id = None
        try:
            body = json.loads(text)
            if not isinstance(body, dict):
                raise ValueError("expected room command")
            kind = body.get("type")
            if kind == "reset":
                self.queue_mic(None)
            elif kind == "vad":
                self.queue_mic(bounded_number(body.get("silence"), 0.5, 2.0, "silence"))
            elif self.participant.role != "owner":
                self.error("forbidden", "owner only")
            elif kind == "config":
                self.room.config(body)
            elif kind == "speak":
                if not valid_id(body.get("id")):
                    raise ValueError("invalid id")
                id = body["id"]
                text = body.get("text")
                if not isinstance(text, str) or len(text) > MAX_SPEAK_TEXT or not text.strip():
                    raise ValueError("invalid text")
                self.room.speak(self, id, text.strip())
            elif kind == "cancel":
                if "id" in body and not valid_id(body["id"]):
                    raise ValueError("invalid id")
                self.room.cancel(body.get("id"))
            else:
                raise ValueError("expected reset, vad, config, speak or cancel")
        except (json.JSONDecodeError, RecursionError):
            self.error("invalid_message", "invalid JSON")
        except ValueError as error:
            self.error("invalid_message", str(error), id)

    async def recognize(self) -> None:
        work = self.room.service.recognize_work
        await self.mic_ready.wait()
        mic = await work(self.room.service.recognize)
        while True:
            if not self.mic:
                self.mic_ready.clear()
                await self.mic_ready.wait()
            message = self.mic.popleft()
            if isinstance(message, bytes):
                # Catch up on queued audio, without waiting for live frames or crossing a control command.
                frames = [message]
                size = len(message)
                while self.mic:
                    following = self.mic[0]
                    if not isinstance(following, bytes) or size + len(following) > MIC_BATCH_BYTES:
                        break
                    self.mic.popleft()
                    frames.append(following)
                    size += len(following)
                self.mic_bytes -= size
                pcm = message if len(frames) == 1 else b"".join(frames)
                for event in await work(mic.feed, pcm):
                    self.send(event)
                    if event["type"] in ("speech_start", "speech_end"):
                        self.room.talking(self, event["type"] == "speech_start")
            elif message is None:
                await work(mic.reset)
                if self.talking:
                    self.send({"type": "speech_end"})
                    self.room.talking(self, False)
                    # A speech_start in flight can arrive after the browser's local reset. Its turn still needs a final.
                    self.send({"type": "final", "text": "", "ms": 0})
            else:
                await work(mic.set_pause, message)
                self.send({"type": "vad", "silence": message})
