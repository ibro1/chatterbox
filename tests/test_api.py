"""API tests against the stub model. Run: CHATTERBOX_FAKE_MODEL=1 python -m pytest tests (no torch or weights needed)."""
import importlib
import io
import sys
import threading
import time
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


def load_app(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("CHATTERBOX_FAKE_MODEL", "1")
    monkeypatch.setenv("CHATTERBOX_API_KEY", KEY)
    monkeypatch.setenv("CHATTERBOX_VOICES_DIR", str(tmp_path / "voices"))
    monkeypatch.setenv("GRADIO_USERNAME", "admin")
    monkeypatch.setenv("GRADIO_PASSWORD", "pw")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("app", None)
    import app
    from tests import fake_chatterbox
    fake_chatterbox.LOADS.clear()
    fake_chatterbox.CALLS.clear()
    return app, fake_chatterbox, TestClient(app.app)


@pytest.fixture
def ctx(monkeypatch, tmp_path):
    return load_app(monkeypatch, tmp_path)


def wav_bytes(seconds, sr=24000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.zeros(int(sr * seconds), dtype="<i2").tobytes())
    return buf.getvalue()


def read_wav(data):
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32767


def speak(client, **body):
    return client.post("/v1/audio/speech", json={"input": "Hello there.", **body}, headers=AUTH)


def test_health_needs_no_auth(ctx):
    app, fake, client = ctx
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "device": "cpu", "model": "turbo", "model_loaded": False}


def test_api_key(ctx):
    app, fake, client = ctx
    body = {"input": "Hello."}
    assert client.post("/v1/audio/speech", json=body).status_code == 401
    assert client.post("/v1/audio/speech", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.post("/v1/audio/speech", json=body, headers={"Authorization": KEY}).status_code == 401
    assert client.get("/v1/voices").status_code == 401
    r = speak(client)
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert client.get("/health").json()["model_loaded"] is True


def test_model_loaded_once_and_peak_normalised(ctx):
    app, fake, client = ctx
    for _ in range(3):
        r = speak(client)
        assert r.status_code == 200
        peak_db = 20 * np.log10(np.max(np.abs(read_wav(r.content))))
        assert -1.6 < peak_db <= -1.4, peak_db  # stub output is 0 dBFS before normalisation
    assert len(fake.LOADS) == 1


def test_chunking_900_chars(ctx):
    app, fake, client = ctx
    sentence = "This sentence is here to pad the text out to a realistic length for a long read."  # 80 chars
    para1 = " ".join([sentence] * 6)
    para2 = " ".join([sentence] * 6)
    text = para1 + "\n\n" + para2
    assert 900 <= len(text) <= 1000
    r = speak(client, input=text)
    assert r.status_code == 200
    chunks = [t for t, _ in fake.CALLS]
    assert all(len(c) <= 280 for c in chunks), [len(c) for c in chunks]
    assert " ".join(chunks) == " ".join(text.split())  # nothing dropped or reordered
    assert all(c.endswith(".") for c in chunks)  # cut on sentence boundaries
    pauses = [p for p, _ in app.split_text(text)]
    assert pauses[0] == 0 and pauses.count(0.6) == 1 and pauses.count(0.25) == len(pauses) - 2
    speech = sum(int(24000 * (0.05 + 0.002 * len(c))) for c in chunks)
    silence = sum(int(24000 * p) for p in pauses)
    assert len(read_wav(r.content)) == speech + silence


def test_over_long_sentence_is_split(ctx):
    app, fake, client = ctx
    words = " ".join(["word"] * 150) + "."  # one 750-char sentence, no commas
    chunks = [c for _, c in app.split_text(words)]
    assert len(chunks) == 3 and all(len(c) <= 280 for c in chunks)
    assert " ".join(chunks) == words


def test_max_chars(monkeypatch, tmp_path):
    app, fake, client = load_app(monkeypatch, tmp_path)
    assert speak(client, input="a" * 3000).status_code == 200
    assert speak(client, input="a" * 3001).status_code == 413
    app, fake, client = load_app(monkeypatch, tmp_path, CHATTERBOX_MAX_CHARS="50")
    assert speak(client, input="a" * 51).status_code == 413


def test_voices(ctx, tmp_path):
    app, fake, client = ctx
    up = lambda name, data: client.post("/v1/voices", data={"name": name}, files={"file": ("v.wav", data, "audio/wav")}, headers=AUTH)
    assert up("narrator", wav_bytes(6)).status_code == 201
    assert (tmp_path / "voices" / "narrator.wav").is_file()
    assert client.get("/v1/voices", headers=AUTH).json() == {"voices": ["default", "narrator"]}
    for bad in ["../evil", "..", "a/b", "/etc/passwd", "x.wav", "", "a" * 65]:
        assert up(bad, wav_bytes(6)).status_code in (400, 422), bad
        assert speak(client, voice=bad).status_code in (400, 422) or bad == "", bad
    assert not any(p.name != "narrator.wav" for p in (tmp_path / "voices").iterdir())
    assert not (tmp_path / "evil.wav").exists()
    assert up("short", wav_bytes(2)).status_code == 400
    assert up("notwav", b"ID3" + b"\0" * 100).status_code == 400

    assert speak(client, voice="narrator").status_code == 200
    assert fake.CALLS[-1][1]["conds"].endswith("narrator.wav")
    assert speak(client, voice="nobody").status_code == 404
    assert speak(client, voice="alloy").status_code == 200  # OpenAI default name -> built-in voice
    assert fake.CALLS[-1][1]["conds"] == "default-voice"  # earlier reference voice did not leak


def test_formats(ctx):
    app, fake, client = ctx
    assert speak(client, response_format="flac").status_code == 400
    r = speak(client, response_format="mp3")
    if app.shutil.which("ffmpeg"):
        assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg" and len(r.content) > 100
    else:
        assert r.status_code == 400


def test_turbo_ignores_unsupported_knobs(ctx):
    app, fake, client = ctx
    speak(client, exaggeration=1.5, cfg_weight=0.9)
    kw = fake.CALLS[-1][1]
    assert (kw["exaggeration"], kw["cfg_weight"], kw["min_p"], kw["top_p"]) == (0.0, 0.0, 0.0, 0.95)
    assert speak(client, language="fr").status_code == 400  # English-only model


def test_multilingual_language(monkeypatch, tmp_path):
    app, fake, client = load_app(monkeypatch, tmp_path, CHATTERBOX_MODEL="multilingual")
    assert speak(client, input="Bonjour.", language="fr").status_code == 200
    assert fake.CALLS[-1][1]["language_id"] == "fr"
    assert fake.LOADS == [("ChatterboxMultilingualTTS", "cpu", {"t3_model": "v3"})]
    assert speak(client, language="xx").status_code == 400


def test_synthesis_is_serialised(ctx):
    app, fake, client = ctx
    active, peak = [0], [0]
    original = fake._FakeTTS.generate

    def slow(self, text, **kw):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        active[0] -= 1
        return original(self, text, **kw)

    fake._FakeTTS.generate = slow
    try:
        threads = [threading.Thread(target=speak, args=(client,)) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
    finally:
        fake._FakeTTS.generate = original
    assert peak[0] == 1 and len(fake.LOADS) == 1


def test_gradio_ui_keeps_its_login(ctx):
    app, fake, client = ctx
    assert client.get("/config").status_code == 401
    assert client.post("/login", data={"username": "admin", "password": "wrong"}).status_code == 400
    assert client.post("/login", data={"username": "admin", "password": "pw"}).status_code == 200
    assert client.get("/config").status_code == 200
