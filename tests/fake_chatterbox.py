"""Test-only stand-in for torch and the Chatterbox models (enabled by CHATTERBOX_FAKE_MODEL=1).

Each generate() call returns a full-scale (0 dBFS) sine wave whose length grows with the text,
so the server's chunking, padding and normalisation can be tested without weights or torch.
"""
import sys
import types

import numpy as np

LOADS = []  # one entry per from_pretrained() call
CALLS = []  # (text, kwargs) per generate() call
SR = 24000


class _Wav:
    def __init__(self, audio):
        self.audio = audio

    def squeeze(self, dim):
        return self

    def numpy(self):
        return self.audio


class _FakeTTS:
    sr = SR

    def __init__(self, name):
        self.name = name
        self.conds = "default-voice"

    @classmethod
    def from_pretrained(cls, device, **kwargs):
        LOADS.append((cls.__name__, device, kwargs))
        return cls(cls.__name__)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5, **kwargs):
        self.conds = wav_fpath

    def generate(self, text, **kwargs):
        CALLS.append((text, dict(kwargs, conds=self.conds)))
        t = np.arange(int(SR * (0.05 + 0.002 * len(text)))) / SR
        return _Wav(np.sin(2 * np.pi * 220 * t).astype(np.float32))


class ChatterboxTTS(_FakeTTS):
    pass


class ChatterboxTurboTTS(_FakeTTS):
    pass


class ChatterboxMultilingualTTS(_FakeTTS):
    pass


def install():
    noop = lambda *a, **k: None
    torch = types.ModuleType("torch")
    torch.manual_seed = noop
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, manual_seed=noop, manual_seed_all=noop)
    modules = {
        "torch": torch,
        "chatterbox": types.ModuleType("chatterbox"),
        "chatterbox.tts": types.ModuleType("chatterbox.tts"),
        "chatterbox.tts_turbo": types.ModuleType("chatterbox.tts_turbo"),
        "chatterbox.mtl_tts": types.ModuleType("chatterbox.mtl_tts"),
    }
    modules["chatterbox.tts"].ChatterboxTTS = ChatterboxTTS
    modules["chatterbox.tts_turbo"].ChatterboxTurboTTS = ChatterboxTurboTTS
    modules["chatterbox.mtl_tts"].ChatterboxMultilingualTTS = ChatterboxMultilingualTTS
    modules["chatterbox.mtl_tts"].SUPPORTED_LANGUAGES = {"en": "English", "fr": "French", "de": "German"}
    sys.modules.update(modules)
