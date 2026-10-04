import hmac
import io
import os
import random
import re
import shutil
import subprocess
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import gradio as gr
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

if os.environ.get("CHATTERBOX_FAKE_MODEL") == "1":
    # Test-only: swaps torch and the chatterbox models for a sine-wave stub, so no weights are needed.
    from tests import fake_chatterbox
    fake_chatterbox.install()

import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = os.environ.get("CHATTERBOX_MODEL", "turbo").lower()  # turbo | nano | standard | multilingual
MULTILINGUAL_VERSION = os.environ.get("CHATTERBOX_MULTILINGUAL_VERSION", "v3")
API_KEY = os.environ.get("CHATTERBOX_API_KEY", "")
VOICES_DIR = Path(os.environ.get("CHATTERBOX_VOICES_DIR", "/app/voices"))
MAX_CHARS = int(os.environ.get("CHATTERBOX_MAX_CHARS", "3000"))
MAX_VOICE_BYTES = 20 * 1024 * 1024
MIN_VOICE_SECONDS = 5.0  # Turbo/Nano refuse shorter reference clips; ~10 s works best for every model
CHUNK_CHARS = 280
SENTENCE_PAUSE = 0.25
PARAGRAPH_PAUSE = 0.6
PEAK_DBFS = -1.5
VOICE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Names OpenAI clients send by default; they fall back to the built-in voice unless a file of that name exists.
OPENAI_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"}

if MODEL_NAME not in ("turbo", "nano", "standard", "multilingual"):
    raise SystemExit(f"CHATTERBOX_MODEL must be turbo, nano, standard or multilingual (got {MODEL_NAME!r})")
TURBO = MODEL_NAME in ("turbo", "nano")  # no exaggeration / CFG / min_p on these
if MODEL_NAME == "multilingual":
    from chatterbox.mtl_tts import SUPPORTED_LANGUAGES
else:
    SUPPORTED_LANGUAGES = {"en": "English"}


# ---------------------------------------------------------------- model

_model = None
_default_conds = None
_model_lock = threading.Lock()
_synth_lock = threading.Lock()  # CPU box: one synthesis at a time, UI and API alike


def load_model():
    if TURBO:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        return ChatterboxTurboTTS.from_pretrained(DEVICE, nano=MODEL_NAME == "nano")
    if MODEL_NAME == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        return ChatterboxMultilingualTTS.from_pretrained(DEVICE, t3_model=MULTILINGUAL_VERSION)
    from chatterbox.tts import ChatterboxTTS
    return ChatterboxTTS.from_pretrained(DEVICE)


def get_model():
    """Load the model once per process; every caller shares it."""
    global _model, _default_conds
    if _model is None:
        with _model_lock:
            if _model is None:
                model = load_model()
                _default_conds = model.conds
                _model = model
    return _model


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def split_long(sentence):
    """Cut one over-long sentence at a comma-like break, else a space, so each piece fits CHUNK_CHARS."""
    while len(sentence) > CHUNK_CHARS:
        cut = max(sentence.rfind(p, 0, CHUNK_CHARS) for p in (", ", "; ", ": ", " — ", "，", "、"))
        cut = cut + 1 if cut > 0 else sentence.rfind(" ", 0, CHUNK_CHARS)
        if cut <= 0:
            cut = CHUNK_CHARS
        yield sentence[:cut].strip()
        sentence = sentence[cut:].strip()
    if sentence:
        yield sentence


def split_text(text):
    """Yield (pause_before_seconds, chunk): whole sentences packed into chunks of <= CHUNK_CHARS."""
    pause = 0.0
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = " ".join(paragraph.split())
        sentences = re.split(r"(?<=[.!?…؟।])\s+|(?<=[。！？])", paragraph)
        chunk = ""
        for piece in (p for s in sentences for p in split_long(s.strip())):
            if chunk and len(chunk) + 1 + len(piece) > CHUNK_CHARS:
                yield pause, chunk
                pause, chunk = SENTENCE_PAUSE, ""
            chunk = f"{chunk} {piece}" if chunk else piece
        if chunk:
            yield pause, chunk
            pause = PARAGRAPH_PAUSE


def normalise(audio):
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio * (10 ** (PEAK_DBFS / 20) / peak)
    return audio.astype(np.float32)


def synthesize(text, voice_path=None, language="en", exaggeration=0.5, temperature=0.8, cfg_weight=0.5,
               seed=0, min_p=0.05, top_p=None, repetition_penalty=1.2):
    """Text of any length -> (sample_rate, peak-normalised float32 mono audio). Raises ValueError on bad input."""
    language = (language or "en").lower()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Language '{language}' is not supported by the {MODEL_NAME} model "
                         f"(supported: {', '.join(SUPPORTED_LANGUAGES)})")
    if TURBO:
        kwargs = dict(temperature=temperature, top_p=top_p or 0.95, repetition_penalty=repetition_penalty,
                      exaggeration=0.0, cfg_weight=0.0, min_p=0.0)
    else:
        kwargs = dict(temperature=temperature, top_p=top_p or 1.0, repetition_penalty=repetition_penalty,
                      exaggeration=exaggeration, cfg_weight=cfg_weight, min_p=min_p)
    if MODEL_NAME == "multilingual":
        kwargs["language_id"] = language

    model = get_model()
    with _synth_lock:
        if seed:
            set_seed(int(seed))
        if voice_path:
            try:
                model.prepare_conditionals(voice_path, exaggeration=kwargs["exaggeration"])
            except AssertionError as e:  # e.g. Turbo: "Audio prompt must be longer than 5 seconds!"
                raise ValueError(str(e) or "Reference audio rejected by the model")
        else:
            model.conds = _default_conds  # a previous reference voice must not leak into this request
        parts = []
        for pause, chunk in split_text(text):
            if pause:
                parts.append(np.zeros(int(model.sr * pause), dtype=np.float32))
            wav = model.generate(chunk, **kwargs)
            parts.append(wav.squeeze(0).numpy().astype(np.float32))
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return model.sr, normalise(audio)


def to_wav(sr, audio):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def to_mp3(wav_bytes):
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "wav", "-i", "pipe:0", "-f", "mp3", "-b:a", "128k", "pipe:1"],
        input=wav_bytes, capture_output=True, check=True,
    )
    return out.stdout


# ---------------------------------------------------------------- UI

def generate(text, audio_prompt_path, language, exaggeration, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty):
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Text is {len(text)} characters; the limit is {MAX_CHARS}.")
    try:
        return synthesize(text, audio_prompt_path, language, exaggeration, temperature, cfgw,
                          seed_num, min_p, top_p, repetition_penalty)
    except ValueError as e:
        raise gr.Error(str(e))


with gr.Blocks(title="Chatterbox TTS") as demo:
    gr.Markdown(f"**Model:** `{MODEL_NAME}` on `{DEVICE}`"
                + (" · supports tags like `[laugh]`, `[chuckle]`, `[cough]`" if TURBO else ""))
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label=f"Text to synthesize (max chars {MAX_CHARS}; long text is read sentence by sentence)",
                max_lines=5
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File (5 s or longer)", value=None)
            language = gr.Dropdown(
                choices=[(f"{name} ({code})", code) for code, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1])],
                value="en", label="Language", visible=MODEL_NAME == "multilingual",
            )
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5, visible=not TURBO)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5, visible=not TURBO)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p || Newer Sampler. Recommend 0.02 > 0.1. Handles Higher Temperatures better. 0.00 Disables", value=0.05, visible=not TURBO)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p || Original Sampler. 1.0 Disables", value=0.95 if TURBO else 1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.2)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    run_btn.click(
        fn=generate,
        inputs=[
            text,
            ref_wav,
            language,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
            min_p,
            top_p,
            repetition_penalty,
        ],
        outputs=audio_output,
    )


# ---------------------------------------------------------------- API

app = FastAPI(title="Chatterbox TTS")


def require_key(authorization: Optional[str] = Header(None)):
    if not API_KEY:
        return
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip().encode(), API_KEY.encode()):
        raise HTTPException(401, "Invalid or missing API key", headers={"WWW-Authenticate": "Bearer"})


def voice_file(name):
    """Map a voice name to a WAV inside VOICES_DIR; names are a strict charset, so no traversal."""
    if not VOICE_NAME.match(name or ""):
        raise HTTPException(400, "Voice names may only use letters, digits, '-' and '_' (max 64)")
    path = (VOICES_DIR / f"{name}.wav").resolve()
    if path.parent != VOICES_DIR.resolve():
        raise HTTPException(400, "Invalid voice name")
    return path


class SpeechRequest(BaseModel):
    input: str
    model: Optional[str] = None  # accepted for OpenAI compatibility; the server model is CHATTERBOX_MODEL
    voice: Optional[str] = None
    response_format: str = "wav"
    speed: Optional[float] = None  # accepted for OpenAI compatibility; Chatterbox has no speed control
    language: str = "en"  # multilingual model only
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    seed: int = 0


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model": MODEL_NAME, "model_loaded": _model is not None}


@app.post("/v1/audio/speech", dependencies=[Depends(require_key)])
def speech(req: SpeechRequest):
    fmt = req.response_format.lower()
    if fmt not in ("wav", "mp3"):
        raise HTTPException(400, "response_format must be 'wav' or 'mp3'")
    if fmt == "mp3" and not shutil.which("ffmpeg"):
        raise HTTPException(400, "mp3 needs ffmpeg, which is not installed on this server; use wav")
    if not req.input.strip():
        raise HTTPException(400, "input is empty")
    if len(req.input) > MAX_CHARS:
        raise HTTPException(413, f"input is {len(req.input)} characters; the limit is {MAX_CHARS}")

    voice_path = None
    if req.voice and req.voice != "default":
        path = voice_file(req.voice)
        if path.is_file():
            voice_path = str(path)
        elif req.voice not in OPENAI_VOICES:
            raise HTTPException(404, f"Unknown voice '{req.voice}'")

    try:
        sr, audio = synthesize(req.input, voice_path, req.language, req.exaggeration,
                               req.temperature, req.cfg_weight, req.seed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    body = to_wav(sr, audio)
    if fmt == "mp3":
        return Response(to_mp3(body), media_type="audio/mpeg")
    return Response(body, media_type="audio/wav")


@app.get("/v1/voices", dependencies=[Depends(require_key)])
def list_voices():
    names = sorted(p.stem for p in VOICES_DIR.glob("*.wav")) if VOICES_DIR.is_dir() else []
    return {"voices": ["default"] + names}


@app.post("/v1/voices", status_code=201, dependencies=[Depends(require_key)])
def upload_voice(name: str = Form(...), file: UploadFile = File(...)):
    path = voice_file(name)
    data = file.file.read(MAX_VOICE_BYTES + 1)
    if len(data) > MAX_VOICE_BYTES:
        raise HTTPException(413, "Reference WAV is larger than 20 MB")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise HTTPException(400, "Reference audio must be a WAV file")
    try:
        with wave.open(io.BytesIO(data)) as w:
            seconds = w.getnframes() / w.getframerate()
    except (wave.Error, EOFError, ZeroDivisionError):
        seconds = None  # float/extensible WAVs the stdlib cannot parse; librosa reads them at synthesis time
    if seconds is not None and seconds < MIN_VOICE_SECONDS:
        raise HTTPException(400, f"Reference clip is {seconds:.1f} s; use at least {MIN_VOICE_SECONDS:.0f} s (about 10 s is best)")
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav.part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return {"name": name}


demo.queue(max_size=50, default_concurrency_limit=1)
GRADIO_AUTH = None
if os.environ.get("GRADIO_USERNAME") and os.environ.get("GRADIO_PASSWORD"):
    GRADIO_AUTH = (os.environ["GRADIO_USERNAME"], os.environ["GRADIO_PASSWORD"])
app = gr.mount_gradio_app(app, demo, path="/", auth=GRADIO_AUTH)


if __name__ == "__main__":
    if not API_KEY:
        print("WARNING: CHATTERBOX_API_KEY is not set; the /v1 endpoints are open to anyone.")
    threading.Thread(target=get_model, daemon=True).start()  # warm up now; early requests wait on the lock
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")),
                proxy_headers=True, forwarded_allow_ips="*")
