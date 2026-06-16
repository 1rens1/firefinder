from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import io
import threading

PORT = 8080


class Output(io.BufferedIOBase):
    def __init__(self):
        self.frame: bytes | None = None
        self.cond = threading.Condition()

    def write(self, buf: bytes) -> int:
        with self.cond:
            self.frame = bytes(buf)
            self.cond.notify_all()
        return len(buf)


output = Output()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b'<html><body><img src="/stream"></body></html>')
            return

        if self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            try:
                while True:
                    with output.cond:
                        output.cond.wait()
                        frame = output.frame

                    if frame is None:
                        continue

                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")

            except (BrokenPipeError, ConnectionResetError):
                return

        self.send_response(404)
        self.end_headers()


picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start_recording(MJPEGEncoder(), FileOutput(output))

print(f"Open: http://0.0.0.0:{PORT}")
ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()