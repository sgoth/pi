"""Exercise /room on loopback, using synthetic microphones and speech (no models or credentials)."""

import asyncio
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from aiohttp import WSMsgType, WSServerHandshakeError, web
from aiohttp.test_utils import TestClient, TestServer

from engines import Engine
from rooms import MAX_MIC_BYTES, MAX_SPEAK_JOBS, MIC_QUEUE_BYTES, MIC_QUEUE_ITEMS, Rooms


class SyntheticMic:
    def __init__(self):
        self.silence = 0.8

    def feed(self, pcm):
        events = []
        for sample in pcm[::2]:
            if sample == 1:
                events.extend([{"type": "speech_start"}, {"type": "interim", "text": "hello", "ms": 1}])
            else:
                events.extend([{"type": "speech_end"}, {"type": "final", "text": str(self.silence), "ms": 1}])
        return events

    def reset(self):
        pass

    def set_pause(self, seconds):
        self.silence = seconds


class SyntheticEngine(Engine):
    id = "synth"
    default_voice = "a"

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.requests = []

    def generate(self, text, voice, speed, stop):
        self.requests.append((text, voice, speed))
        yield np.array([1.0, -1.0], dtype=np.float32)
        if text == "hold":
            self.release.wait(5)
        yield np.array([0.0], dtype=np.float32)


def headers(id="owner", role="owner", room="p2-session-1", name="Owner"):
    return {"X-Room": room, "X-Participant-Id": id, "X-Participant-Name": name, "X-Participant-Role": role}


class RoomTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.loop_thread = threading.get_ident()
        self.engine = SyntheticEngine()
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.rooms = Rooms(self.recognize, self.resolve, {"engine": "synth", "voice": "synth-a", "speed": None}, self.pool)
        app = web.Application()
        app.router.add_get("/room", self.rooms.room)
        app.on_shutdown.append(self.rooms.shutdown)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        self.engine.release.set()
        await self.client.close()
        self.pool.shutdown(wait=True)

    def recognize(self):
        self.assertNotEqual(threading.get_ident(), self.loop_thread)
        return SyntheticMic()

    def resolve(self, engine, voice):
        if engine not in (None, "synth") or voice not in (None, "synth-a", "synth-b"):
            raise ValueError("unknown engine or voice")
        return self.engine, voice[-1] if voice else "a"

    async def join(self, **who):
        socket = await self.client.ws_connect("/room", headers=headers(**who))
        return socket, await self.event(socket, "welcome")

    async def event(self, socket, kind):
        value = await socket.receive_json(timeout=2)
        self.assertEqual(value["type"], kind, value)
        return value

    async def test_admission_and_membership(self):
        for invalid in (headers(room="other"), headers(id="bad.id"), headers(name="%a"), headers(role="admin"),
                        headers(id="owner", role="guest"), {}):
            with self.assertRaises(WSServerHandshakeError) as error:
                await self.client.ws_connect("/room", headers=invalid)
            self.assertEqual(error.exception.status, 400)

        owner, welcome = await self.join()
        self.assertEqual(welcome["tts"], {"engine": "synth", "voice": "synth-a", "speed": None})
        self.assertEqual(welcome["micSampleRate"], 16000)
        self.assertIsNone(welcome["current"])
        guest, joined = await self.join(id="alice", role="guest", name="Al%C3%ADcia")
        self.assertEqual(joined["you"]["name"], "Alícia")
        self.assertEqual(await self.event(owner, "member_joined"), {
            "type": "member_joined", "member": {"id": "alice", "name": "Alícia", "role": "guest", "connections": 1},
        })
        with self.assertRaises(WSServerHandshakeError) as error:
            await self.client.ws_connect("/room", headers=headers(id="alice"))
        self.assertEqual(error.exception.status, 400)

        tab, again = await self.join(id="alice", role="guest", name="Al%C3%ADcia")
        self.assertNotEqual(again["you"]["connectionId"], joined["you"]["connectionId"])
        self.assertEqual([member["connections"] for member in again["members"]], [1, 2])
        self.assertEqual((await self.event(owner, "member_updated"))["member"]["connections"], 2)
        self.assertEqual((await self.event(guest, "member_updated"))["member"]["connections"], 2)
        _, isolated = await self.join(room="p2-session-2")
        self.assertEqual(len(isolated["members"]), 1)
        await tab.close()
        self.assertEqual((await self.event(owner, "member_updated"))["member"]["connections"], 1)
        await guest.close()
        self.assertEqual((await self.event(owner, "member_left"))["member"]["connections"], 0)

    async def test_private_mic_in_socket_order_and_relay_to_other_principals(self):
        owner, _ = await self.join()
        guest, _ = await self.join(id="alice", role="guest", name="Alice")
        tab, _ = await self.join(id="alice", role="guest", name="Alice")
        await self.event(owner, "member_joined")
        await self.event(owner, "member_updated")
        await self.event(guest, "member_updated")

        for command in ({"type": "config", "speed": 1.2}, {"type": "cancel"},
                        {"type": "speak", "id": "abc", "text": "not Pi"}):
            await guest.send_json(command)
            self.assertEqual((await self.event(guest, "error"))["code"], "forbidden")
        await guest.send_bytes(b"\x01")
        self.assertEqual((await self.event(guest, "error"))["code"], "invalid_message")

        await guest.send_json({"type": "vad", "silence": 1.5})
        await guest.send_bytes(b"\x01\0")
        await guest.send_json({"type": "reset"})
        await guest.send_bytes(b"\x01\0")
        await guest.send_bytes(b"\x02\0")
        await self.event(guest, "vad")
        for transcript in ("", "1.5"):
            await self.event(guest, "speech_start")
            self.assertTrue((await self.event(guest, "talking"))["on"])
            await self.event(guest, "interim")
            await self.event(guest, "speech_end")
            self.assertFalse((await self.event(guest, "talking"))["on"])
            self.assertEqual((await self.event(guest, "final"))["text"], transcript)

        frames = []
        talking = []
        for _ in range(7):
            message = await owner.receive(timeout=2)
            if message.type == WSMsgType.BINARY:
                frames.append(message.data)
            else:
                event = message.json()
                self.assertEqual(event["type"], "talking")
                talking.append(event["on"])
        self.assertEqual(frames, [b"\x02\x05alice\0\x01\0", b"\x02\x05alice\0\x01\0", b"\x02\x05alice\0\x02\0"])
        self.assertEqual(talking, [True, False, True, False])
        for on in talking:
            self.assertEqual((await self.event(tab, "talking"))["on"], on)

        await tab.send_bytes(b"\x01\0")
        await tab.send_bytes(b"\x02\0")
        await self.event(tab, "speech_start")
        await self.event(tab, "talking")
        await self.event(tab, "interim")
        await self.event(tab, "speech_end")
        await self.event(tab, "talking")
        self.assertEqual((await self.event(tab, "final"))["text"], "0.8")

    async def test_mic_batches_only_pending_audio_between_commands(self):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        seen = []

        class RecordingMic:
            def feed(self, pcm):
                seen.append(pcm)
                if len(seen) == 1:
                    loop.call_soon_threadsafe(entered.set)
                    if not release.wait(5):
                        raise TimeoutError("test did not release microphone")
                return []

            def reset(self):
                seen.append("reset")

            def set_pause(self, seconds):
                seen.append(seconds)

        self.rooms.recognize = RecordingMic
        owner, _ = await self.join()
        initial = bytes(640)
        frames = [bytes([i, 0]) * 320 for i in range(1, 14)]
        preroll = bytes(MAX_MIC_BYTES)
        try:
            await owner.send_bytes(initial)
            # A single frame reaches the worker without waiting for a full batch.
            await asyncio.wait_for(entered.wait(), timeout=2)
            for frame in frames[:11]:
                await owner.send_bytes(frame)
            await owner.send_json({"type": "reset"})
            await owner.send_bytes(preroll)
            for frame in frames[11:]:
                await owner.send_bytes(frame)
            await owner.send_json({"type": "vad", "silence": 1.2})
            await owner.send_json({"type": "config"})
            await self.event(owner, "config")  # The reader reached the end of the backlog while STT was blocked.
        finally:
            release.set()
        await self.event(owner, "vad")
        self.assertEqual(seen, [initial, b"".join(frames[:10]), frames[10], "reset", preroll,
                                b"".join(frames[11:]), 1.2])

    async def test_mic_queue_bounds_audio_time_and_item_count(self):
        for room, frame, count in (("p2-bytes", bytes(MAX_MIC_BYTES), MIC_QUEUE_BYTES // MAX_MIC_BYTES),
                                   ("p2-items", bytes(2), MIC_QUEUE_ITEMS)):
            owner, _ = await self.join(room=room)
            connection = next(iter(self.rooms.active[room].connections))
            # Fill the queue without running its worker or the separate socket rate limiter.
            for _ in range(count):
                self.assertTrue(connection.queue_mic(frame))
            self.assertFalse(connection.queue_mic(frame))
            closed = await owner.receive(timeout=2)
            self.assertEqual((closed.type, closed.data, closed.extra), (
                WSMsgType.CLOSE, 1013, "Microphone processing is overloaded",
            ))

    async def test_speech_order_config_snapshot_deduplication_and_cancel(self):
        owner, _ = await self.join()
        await owner.send_json({"type": "config", "speed": 0.9, "language": "en"})
        self.assertEqual(await self.event(owner, "config"), {"type": "config", "engine": "synth",
                                                               "voice": "synth-a", "speed": 0.9})
        await owner.send_json({"type": "speak", "id": "one", "text": "hold"})
        self.assertFalse((await self.event(owner, "speak_ack"))["duplicate"])
        await self.event(owner, "tts_start")
        self.assertEqual(await owner.receive_bytes(timeout=2), b"\x01\x03one\0\xff\x7f\x01\x80")

        guest, welcome = await self.join(id="alice", role="guest", name="Alice")
        self.assertEqual(welcome["current"], {"id": "one", "engine": "synth", "voice": "synth-a", "sampleRate": 24000})
        await self.event(owner, "member_joined")
        await owner.send_json({"type": "config", "voice": "synth-b"})
        await owner.send_json({"type": "speak", "id": "two", "text": "next"})
        await owner.send_json({"type": "speak", "id": "two", "text": "duplicate"})
        await owner.send_json({"type": "config", "voice": "synth-a"})
        await owner.send_json({"type": "cancel", "id": "one"})
        await self.event(owner, "config")
        self.assertFalse((await self.event(owner, "speak_ack"))["duplicate"])
        self.assertTrue((await self.event(owner, "speak_ack"))["duplicate"])
        await self.event(owner, "config")
        self.assertEqual(await self.event(owner, "tts_done"), {"type": "tts_done", "id": "one", "cancelled": True})
        await self.event(guest, "config")
        await self.event(guest, "config")
        self.assertTrue((await self.event(guest, "tts_done"))["cancelled"])
        self.engine.release.set()
        for peer in (owner, guest):
            start = await self.event(peer, "tts_start")
            self.assertEqual((start["id"], start["voice"]), ("two", "synth-b"))
            self.assertEqual(await peer.receive_bytes(timeout=2), b"\x01\x03two\0\xff\x7f\x01\x80")
            self.assertEqual(await peer.receive_bytes(timeout=2), b"\x01\x03two\0\0\0")
            self.assertEqual(await self.event(peer, "tts_done"), {"type": "tts_done", "id": "two", "cancelled": False})
        self.assertEqual(self.engine.requests, [("hold", "a", 0.9), ("next", "b", 0.9)])
        await owner.send_json({"type": "speak", "id": "one", "text": "already cancelled"})
        self.assertTrue((await self.event(owner, "speak_ack"))["duplicate"])
        await guest.close()
        await owner.close()
        async with asyncio.timeout(2):
            while self.rooms.active:
                await asyncio.sleep(0)
        _, welcome = await self.join()
        self.assertIsNone(welcome["current"])
        self.assertIsNone(welcome["tts"]["speed"])

    async def test_speech_queue_limit_and_cancel_all(self):
        owner, _ = await self.join()
        await owner.send_json({"type": "speak", "id": "0", "text": "hold"})
        await self.event(owner, "speak_ack")
        await self.event(owner, "tts_start")
        await owner.receive_bytes(timeout=2)
        for index in range(1, MAX_SPEAK_JOBS):
            await owner.send_json({"type": "speak", "id": str(index), "text": "queued"})
            self.assertFalse((await self.event(owner, "speak_ack"))["duplicate"])
        await owner.send_json({"type": "speak", "id": "overflow", "text": "too much"})
        error = await self.event(owner, "error")
        self.assertEqual((error["code"], error["id"]), ("tts_busy", "overflow"))
        await owner.send_json({"type": "cancel"})
        for index in range(MAX_SPEAK_JOBS):
            done = await self.event(owner, "tts_done")
            self.assertEqual((done["id"], done["cancelled"]), (str(index), True))
        self.assertEqual(self.engine.requests, [("hold", "a", 1.0)])


if __name__ == "__main__":
    unittest.main()
