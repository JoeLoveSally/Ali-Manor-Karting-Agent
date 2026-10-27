"""scrcpy 4.1 client: latest decoded frame + single-finger press/release.

Protocol as in Genymobile/scrcpy v4.1 (doc/develop.md): one video socket
(framed H.264 with 12-byte headers) and one control socket. max_size=1600
makes the phone's 1440x3200 screen arrive as 720x1600, the same size as the
recordings in data/raw, so the ground homography applies unchanged.
"""
import hashlib
import json
import queue
import secrets
import shutil
import socket
import struct
import subprocess
import threading
import time
from fractions import Fraction
from pathlib import Path

import av


def _read_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError(f"scrcpy closed after {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def touch_packet(pressed, x, y, width, height):
    # INJECT_TOUCH_EVENT: type, action (0 down / 1 up), pointer id, x, y, w, h, pressure, buttons
    return struct.pack(">BBQiiHHHII", 2, 0 if pressed else 1, 0, x, y, width, height,
                       0xFFFF if pressed else 0, 0, 0)


class Scrcpy:
    def __init__(self, serial, record_dir, max_size=1600, fps=60, touch_xy=(0.86, 0.90)):
        exe = shutil.which("scrcpy")
        self.server = Path(exe).resolve().with_name("scrcpy-server") if exe else None
        self.serial, self.max_size, self.fps, self.touch_xy = serial, max_size, fps, touch_xy
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.frames = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.error = None
        self.proc = self.video = self.control = self.thread = self.port = self.remote = None
        self.width = self.height = 0
        self.pressed = False
        self._lock = threading.Lock()

    def adb(self, *args):
        return subprocess.check_output(["adb", "-s", self.serial, *args], stderr=subprocess.STDOUT,
                                       timeout=15).decode().strip()

    def _deploy_server(self):
        """Put the server jar at self.remote.

        adb push over the WSL USB bridge often stalls, so first look for an identical
        jar already on the device (same md5) and copy it on-device; push otherwise.
        """
        want = hashlib.md5(self.server.read_bytes()).hexdigest()
        listing = self.adb("shell", "md5sum /data/local/tmp/*.jar /data/local/tmp/scrcpy-server* 2>/dev/null || true")
        for line in listing.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == want:
                self.adb("shell", "cp", parts[1], self.remote)
                if self.adb("shell", "md5sum", self.remote).split()[0] == want:
                    return
        self.adb("push", str(self.server), self.remote)

    def __enter__(self):
        try:
            if not (self.server and self.server.is_file()):
                raise FileNotFoundError("scrcpy-server not found next to the scrcpy executable")
            scid = secrets.randbits(30)
            self.remote = f"/data/local/tmp/kart-scrcpy-{scid:x}.jar"
            self._deploy_server()
            self.port = int(self.adb("forward", "tcp:0", f"localabstract:scrcpy_{scid:08x}"))
            log = (self.record_dir / "scrcpy.log").open("w")
            self.proc = subprocess.Popen(
                ["adb", "-s", self.serial, "shell", f"CLASSPATH={self.remote}", "app_process", "/",
                 "com.genymobile.scrcpy.Server", "4.1", f"scid={scid:08x}", "tunnel_forward=true",
                 "audio=false", "control=true", "clipboard_autosync=false", "send_device_meta=false",
                 "send_dummy_byte=true", "send_frame_meta=true", "send_stream_meta=true",
                 f"max_size={self.max_size}", f"max_fps={self.fps}", "video_bit_rate=8000000",
                 "cleanup=true", "power_on=false"],
                stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 10
            while True:
                try:
                    self.video = socket.create_connection(("127.0.0.1", self.port), timeout=1)
                    self.video.settimeout(3)
                    if _read_exact(self.video, 1) == b"\x00":
                        break
                except (OSError, EOFError):
                    if self.video:
                        self.video.close()
                        self.video = None
                    if time.monotonic() > deadline or self.proc.poll() is not None:
                        raise RuntimeError("scrcpy handshake failed, see scrcpy.log")
                    time.sleep(0.1)
            self.control = socket.create_connection(("127.0.0.1", self.port), timeout=2)
            if _read_exact(self.video, 4) != b"h264":
                raise RuntimeError("unexpected codec")
            flags, self.width, self.height = struct.unpack(">III", _read_exact(self.video, 12))
            self.thread = threading.Thread(target=self._decode, daemon=True)
            self.thread.start()
            return self
        except BaseException:
            self.close()
            raise

    def _decode(self):
        dec = av.CodecContext.create("h264", "r")
        dec.thread_count = 1
        index = 0
        try:
            with (self.record_dir / "capture.h264").open("wb") as rec, \
                    (self.record_dir / "frames.jsonl").open("w") as timing:
                while not self.stop.is_set():
                    ts, n = struct.unpack(">QI", _read_exact(self.video, 12))
                    payload = _read_exact(self.video, n)
                    received = time.monotonic()
                    rec.write(payload)
                    if ts & (1 << 62):  # codec config packet
                        dec.extradata = payload
                        continue
                    pkt = av.Packet(payload)
                    pkt.pts = pkt.dts = ts & ((1 << 61) - 1)
                    pkt.time_base = Fraction(1, 1_000_000)
                    for fr in dec.decode(pkt):
                        item = (fr.to_ndarray(format="bgr24"), received, pkt.pts / 1e6)
                        timing.write(json.dumps({"index": index, "pts": item[2],
                                                 "received": received, "decoded": time.monotonic()}) + "\n")
                        timing.flush()
                        index += 1
                        try:
                            self.frames.put_nowait(item)
                        except queue.Full:
                            try:
                                self.frames.get_nowait()
                            except queue.Empty:
                                pass
                            self.frames.put_nowait(item)
        except Exception as exc:  # noqa: BLE001 - surfaced through read()
            if not self.stop.is_set():
                self.error = repr(exc)

    def read(self, timeout=2.0):
        """Newest frame as (bgr image, host receive time, device pts seconds)."""
        if self.error:
            raise RuntimeError(self.error)
        return self.frames.get(timeout=timeout)

    def set_pressed(self, pressed):
        pressed = bool(pressed)
        with self._lock:
            if pressed == self.pressed or not self.control:
                return False
            x = round(self.touch_xy[0] * self.width)
            y = round(self.touch_xy[1] * self.height)
            self.control.sendall(touch_packet(pressed, x, y, self.width, self.height))
            self.pressed = pressed
            return True

    def tap(self, fx, fy, hold=0.08):
        """Tap at a screen fraction (used only for the verified start button)."""
        with self._lock:
            x, y = round(fx * self.width), round(fy * self.height)
            self.control.sendall(touch_packet(True, x, y, self.width, self.height))
            time.sleep(hold)
            self.control.sendall(touch_packet(False, x, y, self.width, self.height))

    def close(self):
        self.stop.set()
        try:
            if self.control and self.pressed:
                self.set_pressed(False)
        except OSError:
            pass
        for s in (self.video, self.control):
            if s:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                s.close()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread:
            self.thread.join(timeout=3)
        if self.port:
            try:
                self.adb("forward", "--remove", f"tcp:{self.port}")
            except subprocess.SubprocessError:
                pass
        if self.remote:
            try:
                self.adb("shell", "rm", "-f", self.remote)
            except subprocess.SubprocessError:
                pass

    def __exit__(self, *exc):
        self.close()
