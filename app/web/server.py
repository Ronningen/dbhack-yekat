import argparse
import asyncio
from collections import defaultdict
import json
import logging
import os
import time
import uuid
import random

import cv2
import streamlink
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRelay
from av import VideoFrame

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
stream_tracks = defaultdict(lambda: defaultdict(list))

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.start_time = int(time.time())

    @property
    def track_id(self):
        return self.track.id

    async def recv(self):
        """
        Processes every frame from `self.track`
        
        Works similar to `map` function over frames

        Returns changed frame
        """
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        cv2.rectangle(img, (100, 170), (380, 230), (150,150,150), -1, cv2.LINE_AA)
        cv2.putText(img,
                            "processing video", (100, 200),
                            0,
                            1,
                            (0, 0, 0),
                            thickness=3,
                            lineType=cv2.LINE_AA)

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        self._persist_analysis(random.randint(0, 1000))

        return new_frame

    def _persist_analysis(self, data):
        """
        Saves `data` in cache by `self.track_id` and timestamp,
        that starts at `self` creation and rounded to 10
        """
        timestamp = round(int(time.time() - self.start_time), -1)
        stream_tracks[self.track_id][timestamp].append(data)


async def index(request):
    """
    Returns index page HTML
    """
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    """
    Returns JavaScript script for index page
    """
    content = open(os.path.join(ROOT, "client.mjs"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    """
    Main controller for establishing `RTCPeerConnection` with clients
    """
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    streams = streamlink.streams(params["stream_link"])  # type: ignore
    stream_url = streams["best"]
    player = MediaPlayer(stream_url.args["url"])


    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                timestamp = round(int(message[4:]) // 1000, -1)
                channel.send(f"pong {timestamp} {stream_tracks[player.video.id][timestamp]}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            pc.addTrack(player.audio)
        elif track.kind == "video":
            pc.addTrack(VideoTransformTrack(player.video))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    assert answer is not None
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):

    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def handle_upload(request):
    reader = await request.multipart()  # создаем объект MultipartReader
    field = await reader.next()  # извлекаем первое поле из объекта MultipartReader

    # проверяем, что поле является файлом
    if field.filename:
        # создаем папку uploads, если ее нет
        if not os.path.exists('uploads'):
            os.mkdir('uploads')

        # сохраняем файл в папку uploads
        with open(os.path.join('uploads', field.filename), 'wb') as f:
            while True:
                # читаем файл частями и записываем его в файловый объект
                chunk = await field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)

        return web.Response(text=f'File {field.filename} has been uploaded')
    else:
        # если поле не является файлом, возвращаем сообщение об ошибке
        return web.Response(text='No file uploaded or invalid field name')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    app = web.Application()
    app.add_routes([web.post('/upload', handle_upload)])
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.mjs", javascript)
    app.router.add_post("/offer", offer)
  
    web.run_app(app,
                access_log=None,
                host=args.host,
                port=args.port)
