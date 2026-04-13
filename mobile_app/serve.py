"""
HAR Activity Tracker server with real CNN inference
===================================================
Serves the app files and runs POST /infer requests
through the trained CNN model (falls back to heuristic if
the checkpoint is missing or PyTorch is unavailable).

Usage:
    python serve.py               # default HTTPS on port 5500
    python serve.py --http        # HTTP mode for Android WebView fallback
    python serve.py --port 9000
    python serve.py --no-browser

On your phone:
  1. Same Wi-Fi as this PC
  2. Open the Network URL printed below
  3. Tap "Advanced -> Proceed" on the cert warning
  4. Tap START  (iOS: also tap Enable Sensors)

Requires:
    pip install cryptography        (HTTPS cert generation, not needed for --http)
    pip install scipy               (window resampling for inference)
"""

import http.server
import socketserver
import socket
import ssl
import os
import sys
import json
import time
import argparse
import webbrowser
import threading
import tempfile
import ipaddress
import datetime
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
MOBILE_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = MOBILE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

ACTIVITY_NAMES = ['Lying', 'Sitting', 'Standing', 'Walking', 'Running', 'Cycling', 'Stairs']

# ── Model (loaded once at startup) ────────────────────────────
_model      = None
_model_mode = 'heuristic'   # 'model' or 'heuristic'
_model_lock = threading.Lock()


def _load_model():
    global _model, _model_mode
    try:
        import torch
        from src.models.cnn import CNN1D
        ckpt = PROJECT_ROOT / 'results' / 'checkpoints' / 'cnn_best.pt'
        if not ckpt.exists():
            print(f'  [infer] checkpoint not found: {ckpt.name}  -> heuristic mode')
            return
        data = torch.load(ckpt, map_location='cpu', weights_only=False)
        m = CNN1D()
        m.load_state_dict(data['model_state_dict'])
        m.eval()
        with _model_lock:
            _model      = m
            _model_mode = 'model'
        print(f'  [infer] CNN model loaded ({ckpt.name})')
    except Exception as e:
        print(f'  [infer] model load failed: {e}  -> heuristic mode')


# ── Preprocessing ──────────────────────────────────────────────
def _preprocess(samples):
    from scipy.signal import resample
    arr = np.array(samples, dtype=np.float32)   # (N, 6)
    w   = resample(arr, 256).astype(np.float32) # (256, 6)
    rng = np.random.default_rng(42)
    ankle = w + rng.normal(0, 0.04, w.shape).astype(np.float32)
    w12 = np.concatenate([w, ankle], axis=1)    # (256, 12)
    mu  = w12.mean(axis=0, keepdims=True)
    sd  = w12.std(axis=0,  keepdims=True) + 1e-8
    return ((w12 - mu) / sd)[np.newaxis]        # (1, 256, 12)


# ── Inference ──────────────────────────────────────────────────
def _model_infer(samples):
    import torch
    t = torch.from_numpy(_preprocess(samples))
    with _model_lock:
        with torch.no_grad():
            p = torch.softmax(_model(t), dim=-1).squeeze().numpy()
    return int(p.argmax()), p.tolist()


def _mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def _std(values):
    return float(np.std(values)) if len(values) else 0.0


def _zero_crossing_frequency(values, sample_rate=50.0):
    if len(values) < 2:
        return 0.0
    crossings = int(np.sum(np.diff(np.sign(values)) != 0))
    return crossings / (2 * (len(values) / sample_rate))


def _window_stats(samples):
    arr = np.array(samples, dtype=np.float32)
    accel = arr[:, :3]
    gyro = arr[:, 3:6]

    accel_mean = accel.mean(axis=0, keepdims=True)
    lin = accel - accel_mean

    gravity_abs = np.abs(accel_mean.squeeze())
    gravity_total = float(np.sum(gravity_abs) + 1e-8)
    gravity_shares = gravity_abs / gravity_total

    lin_mag = np.sqrt(np.sum(lin ** 2, axis=1))
    raw_mag = np.sqrt(np.sum(accel ** 2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=1))

    return {
        'lin_rms': float(np.sqrt(np.mean(lin_mag ** 2))),
        'lin_std': _std(lin_mag),
        'raw_std': _std(raw_mag),
        'gyro_mean': _mean(gyro_mag),
        'gravity_shares': gravity_shares,
        'dominant_gravity_axis': ['ax', 'ay', 'az'][int(np.argmax(gravity_shares))],
        'step_freq': max(
            _zero_crossing_frequency(lin[:, 0]),
            _zero_crossing_frequency(lin[:, 1]),
            _zero_crossing_frequency(lin[:, 2]),
        ),
    }


def _heuristic(samples):
    stats = _window_stats(samples)
    s    = np.full(7, 0.05)
    if stats['lin_rms'] < 0.15 and stats['gyro_mean'] < 0.08 and stats['raw_std'] < 0.10:
        if stats['dominant_gravity_axis'] != 'ay' and stats['gravity_shares'][2] > 0.34:
            s[0] += 2.8; s[1] += 0.3
        else:
            s[2] += 2.2; s[1] += 1.1
    elif stats['lin_rms'] < 0.45 and stats['gyro_mean'] < 0.22:
        if stats['dominant_gravity_axis'] == 'ay' and stats['gravity_shares'][1] > 0.42:
            s[2] += 2.0; s[1] += 1.1
        else:
            s[1] += 1.4; s[2] += 1.0; s[0] += 0.2
    elif 1.15 <= stats['step_freq'] <= 2.8 and 0.25 <= stats['lin_rms'] < 4.5:
        s[3] += 3.0
        if stats['step_freq'] > 2.35 or stats['lin_rms'] > 2.2:
            s[4] += 0.8
    elif stats['lin_rms'] >= 3.0 and stats['step_freq'] > 2.2:
        s[4] += 2.8; s[3] += 0.5
    elif stats['gyro_mean'] > 0.45 and stats['lin_rms'] < 3.5:
        s[5] += 2.1; s[3] += 0.4
    elif stats['lin_std'] > 0.9 and 1.0 < stats['step_freq'] < 2.4:
        s[6] += 2.3; s[3] += 0.5
    else:
        s[2] += 1.4
    if stats['gyro_mean'] > 0.3:
        s[5] += 0.4; s[6] += 0.2
    if stats['step_freq'] > 0.9:
        s[3] += 0.2
    if stats['lin_rms'] < 0.15 and stats['gyro_mean'] < 0.08 and stats['dominant_gravity_axis'] != 'ay':
        s[0] += 0.5
    e = np.exp(s - s.max()); p = e / e.sum()
    return int(p.argmax()), p.tolist()


# ── HTTP handler ───────────────────────────────────────────────
class HARHandler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(MOBILE_DIR), **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin',  '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        if self.path != '/infer':
            self.send_response(404)
            self.end_headers()
            return
        t0 = time.perf_counter()
        try:
            n       = int(self.headers.get('Content-Length', 0))
            body    = self.rfile.read(n)
            payload = json.loads(body)
            samples = payload.get('window', [])
            if len(samples) < 10:
                raise ValueError(f'Need >= 10 samples, got {len(samples)}')
            heur_act, heur_conf = _heuristic(samples)
            if _model is not None:
                model_act, model_conf = _model_infer(samples)
                model_best = model_conf[model_act]
                heur_best = heur_conf[heur_act]

                if heur_act in (3, 4) and heur_best >= 0.40 and (model_act not in (3, 4) or heur_best > model_best + 0.05):
                    act, conf, src = heur_act, heur_conf, 'heuristic'
                else:
                    blend = (0.65 * np.array(model_conf)) + (0.35 * np.array(heur_conf))
                    act, conf = int(np.argmax(blend)), blend.tolist()
                    src = 'hybrid'
            else:
                act, conf, src = heur_act, heur_conf, 'heuristic'
            out = json.dumps({
                'activity':    act,
                'confidences': conf,
                'source':      src,
                'samples':     len(samples),
                'latency_ms':  round((time.perf_counter() - t0) * 1000, 1),
            }).encode()
            self.send_response(200)
            self.send_header('Content-Type',   'application/json')
            self.send_header('Content-Length', str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        except Exception as e:
            err = json.dumps({'error': str(e)}).encode()
            self.send_response(500)
            self.send_header('Content-Type',   'application/json')
            self.send_header('Content-Length', str(len(err)))
            self.end_headers()
            self.wfile.write(err)

    def log_message(self, fmt, *args):
        code = str(args[1]) if len(args) > 1 else ''
        if code not in ('200', '304', '206', '204'):
            sys.stdout.write(f'  [{code}] {self.path}\n')
            sys.stdout.flush()


# ── TLS ────────────────────────────────────────────────────────
def _make_cert(ip):
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        print('  ERROR: run:  pip install cryptography'); sys.exit(1)

    key  = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u'HAR')])
    san  = x509.SubjectAlternativeName([
        x509.DNSName(u'localhost'),
        x509.IPAddress(ipaddress.IPv4Address('127.0.0.1')),
        x509.IPAddress(ipaddress.IPv4Address(ip)),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
        .add_extension(san, critical=False)
        .sign(key, hashes.SHA256())
    )
    cf = tempfile.NamedTemporaryFile(delete=False, suffix='.pem')
    cf.write(cert.public_bytes(serialization.Encoding.PEM)); cf.close()
    kf = tempfile.NamedTemporaryFile(delete=False, suffix='.pem')
    kf.write(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )); kf.close()
    return cf.name, kf.name


# ── Helpers ────────────────────────────────────────────────────
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: s.connect(('8.8.8.8', 80)); return s.getsockname()[0]
    except: return '127.0.0.1'
    finally: s.close()


def find_free_port(start):
    for p in range(start, start + 20):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(('0.0.0.0', p)); s.close(); return p
        except OSError:
            s.close()
    print(f'  ERROR: no free port in {start}-{start+19}'); sys.exit(1)


def print_qr(url):
    try:
        import qrcode
        qr = qrcode.QRCode(border=2)
        qr.add_data(url); qr.make(fit=True); print(); qr.print_ascii(invert=True)
    except ImportError:
        pass


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',       type=int, default=5500)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--http',       action='store_true', help='Serve over HTTP (no TLS)')
    args = parser.parse_args()

    port = find_free_port(args.port)
    ip   = get_local_ip()

    # Load model in background (instant startup)
    threading.Thread(target=_load_model, daemon=True).start()

    scheme = 'http' if args.http else 'https'
    cert = key = None
    if not args.http:
        cert, key = _make_cert(ip)

    local_url   = f'{scheme}://localhost:{port}'
    network_url = f'{scheme}://{ip}:{port}'

    print()
    print('  +--------------------------------------------------+')
    print('  |  HAR Activity Tracker  -  Live Demo Server       |')
    print('  +--------------------------------------------------+')
    print()
    print(f'  Local:    {local_url}')
    print(f'  Network:  {network_url}')
    print()
    print_qr(network_url)
    print()
    print('  To install as a phone app:')
    print('  1. Open the Network URL on your phone')
    if args.http:
        print('  2. Using HTTP mode (no certificate prompt)')
    else:
        print('  2. Accept the certificate warning (Advanced -> Proceed)')
    print('  3. Android Chrome: tap menu (3 dots) -> "Add to Home screen"')
    print('     iOS Safari:      tap Share -> "Add to Home Screen"')
    print('  4. Open the installed app icon -- it runs full-screen')
    print()
    print('  Inference: model loads in background (check [infer] log)')
    print('  Ctrl+C to stop.')
    print()

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(local_url)).start()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('0.0.0.0', port), HARHandler) as httpd:
        if not args.http:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)
            httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\n  Server stopped.')
        finally:
            if cert and os.path.exists(cert):
                os.unlink(cert)
            if key and os.path.exists(key):
                os.unlink(key)


if __name__ == '__main__':
    main()
