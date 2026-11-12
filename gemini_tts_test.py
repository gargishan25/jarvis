"""Try Gemini text-to-speech voices from the Terminal. Read-only test: changes nothing.

  python gemini_tts_test.py voices en-IN
  python gemini_tts_test.py voices en-US warm
  python gemini_tts_test.py say VOICE_ID
  python gemini_tts_test.py say VOICE_ID "Hi, it's JARVIS." gemini-3.8-flash-tts
"""
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

KEY = os.environ.get("GEMINI_API_KEY")
BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.8-flash-lite-tts"
DEFAULT_TEXT = "Hi, it's JARVIS. Everything looks good today. How can I help?"


def call(method, path, body=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"x-goog-api-key": KEY, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode()[:800]}")


def pick(d, *names):
    for n in names:
        if d.get(n) not in (None, ""):
            return d[n]
    return ""


def list_voices(lang, search):
    params = {"type": "prebuilt", "page_size": 100}
    if lang and lang.lower() != "all":
        params["language_code"] = lang
    if search:
        params["search"] = search
    data = call("GET", "/voices?" + urllib.parse.urlencode(params))
    voices = data.get("voices", [])
    if not voices:
        print("No voices returned. Raw response:\n" + json.dumps(data)[:600])
    for v in voices:
        print(
            f"{pick(v, 'id', 'name')} | {pick(v, 'display_name', 'displayName')} | "
            f"{pick(v, 'language_code', 'languageCode')} {pick(v, 'accent')} "
            f"{pick(v, 'gender')} pitch={pick(v, 'pitch')} | {pick(v, 'description')}"
        )
    print(f"\n{len(voices)} voices.")


def say(voice, text, model):
    body = {
        "model": model,
        "input": [{"type": "user_input", "content": [{"type": "text", "text": text}]}],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": [{"voice": voice}]},
    }
    r = call("POST", "/interactions", body)
    audio = [
        c
        for s in r.get("steps", [])
        if s.get("type") == "model_output"
        for c in s.get("content", [])
        if c.get("type") == "audio"
    ]
    if not audio:
        sys.exit("No audio came back. Raw response:\n" + json.dumps(r)[:800])
    with open("tts_test.wav", "wb") as f:
        f.write(base64.b64decode(audio[-1]["data"]))
    print(f"Saved tts_test.wav ({model}, voice {voice}). Playing...")
    subprocess.run(["afplay", "tts_test.wav"])


def main():
    if not KEY:
        sys.exit("GEMINI_API_KEY is not set in this Terminal.")
    a = sys.argv[1:]
    if a and a[0] == "voices":
        list_voices(a[1] if len(a) > 1 else "en-US", a[2] if len(a) > 2 else "")
    elif len(a) >= 2 and a[0] == "say":
        say(a[1], a[2] if len(a) > 2 else DEFAULT_TEXT, a[3] if len(a) > 3 else DEFAULT_MODEL)
    else:
        print(__doc__)


main()
