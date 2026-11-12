"""Free JARVIS: works with Gemini's free tier or a local Ollama model.

Tools: shell (approval-gated), notes memory, GitHub (read API + one-approval PR tool),
local coding on a scratch clone (edit, run tests, open PR), read-only Google Calendar. Voice: Kokoro/say speech out,
Whisper speech in (/talk), and a "Hey Jarvis" wake word (/listen).

Run it with the `jarvis` command you set up in ~/.zshrc, or:
    cd ~/Documents/J.A.R.V.I.S && source .venv/bin/activate && python3 jarvis.py

Override the model any time with JARVIS_MODEL="model-name".
Use a local model with JARVIS_PROVIDER=ollama.
"""
import base64
import collections
import datetime
import importlib.util
import json
import os
import pathlib
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    import numpy as np
except ImportError:
    np = None
try:
    import sounddevice as sd
except Exception:  # missing package or no audio library
    sd = None

try:
    import truststore

    truststore.inject_into_ssl()  # use the Mac's trusted certificates (helps with Umbrella)
except ImportError:
    pass

from openai import OpenAI

PROVIDER = os.environ.get("JARVIS_PROVIDER", "gemini")

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "model": "gemini-3.5-flash-lite",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "qwen2.5:7b",
    },
}

cfg = PROVIDERS[PROVIDER]
MODEL = os.environ.get("JARVIS_MODEL", cfg["model"])
client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=30, max_retries=1)

TIMING = os.environ.get("JARVIS_TIMING", "1") != "0"  # prints [time] lines; JARVIS_TIMING=0 hides them
THINKING = os.environ.get("JARVIS_THINKING", "low").strip().lower()  # Gemini thinking effort
if THINKING in ("off", "none", "0", "false"):
    THINKING = ""
_thinking_ok = True
_clock = {"t0": 0.0}


def tlog(label, secs):
    if TIMING:
        print(f"  [time] {label}: {secs:.1f}s")


def chat_call(messages):
    """One model call. Asks Gemini for low thinking effort (faster); if the API rejects it, retry without."""
    global _thinking_ok
    kwargs = dict(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt()}] + messages,
        tools=TOOLS,
    )
    if PROVIDER == "gemini" and THINKING and _thinking_ok:
        try:
            return client.chat.completions.create(reasoning_effort=THINKING, **kwargs)
        except Exception as e:
            if getattr(e, "status_code", None) != 400:
                raise
            _thinking_ok = False
            print(f"  [note: the model rejected reasoning_effort='{THINKING}', so it is turned off]")
    return client.chat.completions.create(**kwargs)


JARVIS_DIR = pathlib.Path.home() / "jarvis"
NOTES = JARVIS_DIR / "notes"
NOTES.mkdir(parents=True, exist_ok=True)

CAL_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
def _load_personal():
    """Private settings (name, slang, allowed repos) live OUTSIDE the project, in ~/jarvis/personal.json."""
    try:
        return json.loads((JARVIS_DIR / "personal.json").read_text())
    except (OSError, ValueError):
        return {}


PERSONAL = _load_personal()
ALLOWED_REPOS = {  # lowercase "owner/repo". Add yours in personal.json or JARVIS_ALLOWED_REPOS="owner/repo,..."
    r.strip().lower()
    for r in list(PERSONAL.get("allowed_repos", [])) + os.environ.get("JARVIS_ALLOWED_REPOS", "").split(",")
    if r.strip()
}
NAME_SUFFIX = f", {PERSONAL['name']}" if PERSONAL.get("name") else ""
WORK = JARVIS_DIR / "work"  # scratch clones live here


def fn(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


TOOLS = [
    fn(
        "run_shell",
        "Run a shell command on the user's Mac (git, python, ls, etc). The user must approve each command.",
        {"command": {"type": "string"}},
        ["command"],
    ),
    fn(
        "save_note",
        "Save or overwrite a note in long-term memory: user preferences, goals, project context.",
        {"name": {"type": "string"}, "content": {"type": "string"}},
        ["name", "content"],
    ),
    fn("read_note", "Read a note from long-term memory.", {"name": {"type": "string"}}, ["name"]),
    fn("list_notes", "List all saved notes."),
    fn(
        "github_api",
        "Call the GitHub REST API. Use this for READING (GET): listing repos, reading files, checking PRs. method is GET, POST, PUT, PATCH or DELETE. path starts with / (e.g. /user/repos). body is an optional JSON string. Non-GET calls need user approval. To create or change files, use github_open_pr instead.",
        {"method": {"type": "string"}, "path": {"type": "string"}, "body": {"type": "string"}},
        ["method", "path"],
    ),
    fn(
        "github_open_pr",
        "Create or update one or more files in a GitHub repo on a NEW branch and open a pull request, all in one step (the user approves once after seeing the files). files is a list of objects with path (e.g. 'hello.py') and content (the plain text of the whole file, NOT base64). base is optional and defaults to the repo's default branch.",
        {
            "repo": {"type": "string", "description": "OWNER/REPO"},
            "branch": {"type": "string", "description": "new branch name, e.g. add-hello"},
            "title": {"type": "string"},
            "body": {"type": "string", "description": "PR description"},
            "base": {"type": "string"},
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        ["repo", "branch", "title", "files"],
    ),
    fn(
        "calendar_events",
        "Read events from the user's primary Google Calendar (read-only). start and end are ISO 8601 timestamps WITH a timezone offset, e.g. 2026-11-12T00:00:00-08:00. Returns one line per event.",
        {"start": {"type": "string"}, "end": {"type": "string"}},
        ["start", "end"],
    ),
    fn(
        "work_clone",
        "Make a fresh local clone of an allowed GitHub repo in the scratch folder, discarding any earlier local edits. Always do this first for a coding task.",
        {"repo": {"type": "string", "description": "OWNER/REPO"}},
        ["repo"],
    ),
    fn(
        "work_read",
        "Read a file, or list a folder, in the local clone. path is relative to the repo root; an empty path lists the root.",
        {"repo": {"type": "string"}, "path": {"type": "string"}},
        ["repo"],
    ),
    fn(
        "work_write",
        "Create or overwrite ONE file in the local clone with its full plain text. This only changes the local scratch copy, not GitHub.",
        {"repo": {"type": "string"}, "path": {"type": "string"}, "content": {"type": "string"}},
        ["repo", "path", "content"],
    ),
    fn(
        "work_test",
        "Run the repo's tests (python -m pytest) in the local clone. The user must approve.",
        {"repo": {"type": "string"}},
        ["repo"],
    ),
    fn(
        "work_open_pr",
        "Open a pull request containing every change made in the local clone. Shows the diff and needs one user approval. branch must be a NEW branch name.",
        {
            "repo": {"type": "string"},
            "branch": {"type": "string"},
            "title": {"type": "string"},
            "body": {"type": "string", "description": "PR description"},
        },
        ["repo", "branch", "title"],
    ),
]


def note_path(name):
    safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip() or "untitled"
    return NOTES / f"{safe}.md"


def gh_request(method, path, body=None):
    """Call the GitHub API. Returns (status_code, parsed_json_or_None)."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set.")
    req = urllib.request.Request(
        "https://api.github.com" + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "jarvis",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read().decode())
        except Exception:
            data = None
        return e.code, data


def open_pr(args):
    """Create a branch, commit the files, and open a PR. One approval, shown up front."""
    try:
        repo = args["repo"].strip()
        branch = args["branch"].strip()
        title = args["title"]
        pr_body = args.get("body", "")
        files = args["files"]
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            return "Invalid repo. Use OWNER/REPO."
        if not re.fullmatch(r"[A-Za-z0-9._/-]+", branch):
            return "Invalid branch name. Use letters, numbers, dots, dashes, slashes."
        if repo.lower() not in ALLOWED_REPOS:
            return f"Refusing: {repo} is not in the allowed repo list. Set JARVIS_ALLOWED_REPOS=owner/repo to allow it."
        if branch.lower() in ("main", "master"):
            return "Refusing: never open a PR from main or master."
        if not files:
            return "No files given."
        paths = [f["path"] for f in files]
        if len(paths) != len(set(paths)):
            return "Duplicate file paths in one PR are not supported."
        for f in files:
            p = f["path"]
            if p.startswith("/") or ".." in p.split("/") or p.startswith(".github/"):
                return f"Refusing to write path: {p}"

        status, info = gh_request("GET", f"/repos/{repo}")
        if status != 200:
            return f"Cannot read repo (HTTP {status}): {info}"
        base = args.get("base") or info["default_branch"]
        if branch == base:
            return "The new branch must be different from the base branch."

        # Work out which files are new and which already exist (so we can update them).
        existing = {}
        for f in files:
            qpath = urllib.parse.quote(f["path"], safe="/")
            st, data = gh_request(
                "GET", f"/repos/{repo}/contents/{qpath}?ref={urllib.parse.quote(base, safe='')}"
            )
            existing[f["path"]] = data["sha"] if st == 200 and isinstance(data, dict) else None

        print(f"\n[JARVIS wants to open a pull request]")
        print(f"  Repo:   {repo}")
        print(f"  Branch: {branch}  ->  {base}")
        print(f"  Title:  {title}")
        for f in files:
            label = "UPDATE (overwrites existing file)" if existing[f["path"]] else "NEW"
            lines = f["content"].splitlines()
            print(f"\n  --- {f['path']} [{label}] ({len(lines)} lines) ---")
            for line in lines[:30]:
                print(f"  | {line}")
            if len(lines) > 30:
                print(f"  | ... ({len(lines) - 30} more lines)")
        if not approve("\nApprove? (y/n): "):
            return "User denied this pull request."

        st, ref = gh_request(
            "GET", f"/repos/{repo}/git/ref/heads/{urllib.parse.quote(base, safe='/')}"
        )
        if st != 200:
            return f"Could not read base branch '{base}' (HTTP {st}): {ref}"
        sha = ref["object"]["sha"]

        st, out = gh_request(
            "POST", f"/repos/{repo}/git/refs", {"ref": f"refs/heads/{branch}", "sha": sha}
        )
        if st != 201:
            return f"Could not create branch (HTTP {st}): {out}. Try a different branch name."

        for f in files:
            qpath = urllib.parse.quote(f["path"], safe="/")
            payload = {
                "message": f"{'Update' if existing[f['path']] else 'Add'} {f['path']}",
                "content": base64.b64encode(f["content"].encode()).decode(),
                "branch": branch,
            }
            if existing[f["path"]]:
                payload["sha"] = existing[f["path"]]
            st, out = gh_request("PUT", f"/repos/{repo}/contents/{qpath}", payload)
            if st not in (200, 201):
                return (
                    f"Failed writing {f['path']} (HTTP {st}): {out}. "
                    f"The branch '{branch}' was already created."
                )

        st, pr = gh_request(
            "POST",
            f"/repos/{repo}/pulls",
            {"title": title, "head": branch, "base": base, "body": pr_body},
        )
        if st != 201:
            return f"Files were committed to '{branch}' but opening the PR failed (HTTP {st}): {pr}"
        return f"Opened pull request: {pr['html_url']}"
    except Exception as e:
        return f"Error: {e}"


def work_dir(repo):
    repo = repo.strip()
    if repo.lower() not in ALLOWED_REPOS:
        raise ValueError(f"{repo} is not in the allowed repo list. Set JARVIS_ALLOWED_REPOS=owner/repo to allow it.")
    return WORK / repo.lower().split("/")[1]


def safe_path(base, rel):
    """Resolve rel inside base; refuse anything that escapes it or touches .git."""
    base = base.resolve()
    p = (base / rel).resolve()
    if not p.is_relative_to(base):
        raise ValueError("Path escapes the work folder.")
    if ".git" in p.relative_to(base).parts:
        raise ValueError("Refusing to touch .git.")
    return p


def git(cwd, *args):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


def work_clone(args):
    repo = args["repo"].strip()
    d = work_dir(repo)
    WORK.mkdir(parents=True, exist_ok=True)
    if d.exists():
        shutil.rmtree(d)
    print(f"\n[JARVIS is cloning]: {repo} -> {d}")
    r = git(WORK, "clone", "--depth", "1", f"https://github.com/{repo}.git", d.name)
    if r.returncode != 0:
        return f"Clone failed: {(r.stdout + r.stderr)[-1500:]}"
    return f"Cloned {repo} into {d}"


def work_read(args):
    d = work_dir(args["repo"])
    if not d.exists():
        return "Not cloned yet. Call work_clone first."
    p = safe_path(d, args.get("path") or ".")
    if not p.exists():
        return "No such file or folder."
    if p.is_dir():
        items = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir() if x.name != ".git")
        return "\n".join(items) or "(empty folder)"
    return p.read_text(errors="replace")[:20000]


def work_write(args):
    d = work_dir(args["repo"])
    if not d.exists():
        return "Not cloned yet. Call work_clone first."
    p = safe_path(d, args["path"])
    content = args["content"]
    if len(content) > 100_000:
        return "Refusing: file is too large."
    existed = p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    print(f"\n[JARVIS wrote locally]: {args['path']} ({len(content.splitlines())} lines)")
    return f"{'Updated' if existed else 'Created'} {args['path']} in the local copy. Nothing is on GitHub yet."


def work_test(args):
    d = work_dir(args["repo"])
    if not d.exists():
        return "Not cloned yet. Call work_clone first."
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    print(f"\n[JARVIS wants to run tests]: {' '.join(cmd)}")
    print(f"  in {d}")
    print("  (this runs code from the repo on your Mac; keys and tokens are removed from its environment)")
    if not approve():
        return "User denied running tests."
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(s in k.upper() for s in ("TOKEN", "KEY", "SECRET", "PASSWORD"))
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=120, env=env)
    except subprocess.TimeoutExpired:
        return "Tests timed out after 120s."
    result = (r.stdout + r.stderr)[-6000:]
    print(f"\n[pytest exit code {r.returncode}]\n{result}")
    return f"pytest exit code {r.returncode} (0 = all passed, 5 = no tests found)\n" + result

def work_open_pr(args):
    repo = args["repo"].strip()
    d = work_dir(repo)
    if not d.exists():
        return "Not cloned yet. Call work_clone first."
    r = git(d, "status", "--porcelain", "-z", "--untracked-files=all")
    if r.returncode != 0:
        return f"git status failed: {r.stderr[-500:]}"
    files, skipped = [], []
    for entry in r.stdout.split("\0"):
        if len(entry) < 4:
            continue
        xy, path = entry[:2], entry[3:]
        if "R" in xy or "C" in xy:
            return "Renames/copies are not supported. Use plain edits."
        if "D" in xy:
            skipped.append(f"{path} (deleted; deletions are not supported)")
            continue
        if "__pycache__" in path or path.endswith(".pyc") or path.startswith(".pytest_cache/"):
            continue
        try:
            files.append({"path": path, "content": (d / path).read_text()})
        except (UnicodeDecodeError, OSError):
            skipped.append(f"{path} (not readable UTF-8 text)")
    if skipped:
        print("\n[Skipped, will NOT be in the PR]: " + "; ".join(skipped))
    if not files:
        return "No changes to submit." + (f" Skipped: {'; '.join(skipped)}" if skipped else "")
    paths = [f["path"] for f in files]
    git(d, "add", "-N", "--", *paths)  # lets `git diff` show brand-new files too
    diff = git(d, "diff", "--", *paths).stdout.splitlines()
    print("\n[Changes compared with the repo's current default branch]")
    for line in diff[:200]:
        print(f"  {line}")
    if len(diff) > 200:
        print(f"  ... ({len(diff) - 200} more diff lines)")
    return open_pr(
        {
            "repo": repo,
            "branch": args["branch"],
            "title": args["title"],
            "body": args.get("body", ""),
            "files": files,
        }
    )


def run_work_tool(name, args):
    try:
        if name == "work_clone":
            return work_clone(args)
        if name == "work_read":
            return work_read(args)
        if name == "work_write":
            return work_write(args)
        if name == "work_test":
            return work_test(args)
        if name == "work_open_pr":
            return work_open_pr(args)
        return "Unknown tool."
    except Exception as e:
        return f"Error: {e}"


def calendar_token():
    """Return a valid Google access token, logging in through the browser if needed."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_file = JARVIS_DIR / "calendar_token.json"
    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), CAL_SCOPES)
    if creds and not creds.valid and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            creds = None
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(JARVIS_DIR / "credentials.json"), CAL_SCOPES
        )
        creds = flow.run_local_server(port=0)
    token_file.write_text(creds.to_json())
    token_file.chmod(0o600)
    return creds.token


def run_tool(name, args):
    if name.startswith("work_"):
        return run_work_tool(name, args)
    if name == "run_shell":
        cmd = args["command"]
        print(f"\n[JARVIS wants to run]: {cmd}")
        if not approve():
            return "User denied this command."
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
            return (r.stdout + r.stderr)[-8000:] or "(no output)"
        except subprocess.TimeoutExpired:
            return "Command timed out after 120s."
    if name == "save_note":
        note_path(args["name"]).write_text(args["content"])
        return "Saved."
    if name == "read_note":
        p = note_path(args["name"])
        return p.read_text() if p.exists() else "No such note."
    if name == "list_notes":
        return "\n".join(p.stem for p in NOTES.glob("*.md")) or "(no notes yet)"
    if name == "github_api":
        method = args["method"].upper()
        path = args["path"] if args["path"].startswith("/") else "/" + args["path"]
        body = args.get("body")
        print(f"\n[JARVIS is calling GitHub]: {method} {path}")
        if method != "GET":
            print(f"\n[JARVIS wants to call GitHub]: {method} {path}\n{body or ''}")
            if not approve():
                return "User denied this call."
        try:
            parsed = json.loads(body) if body else None
            status, data = gh_request(method, path, parsed)
            return f"HTTP {status}: " + json.dumps(data)[:8000]
        except Exception as e:
            return f"Error: {e}"
    if name == "github_open_pr":
        return open_pr(args)
    if name == "calendar_events":
        print(f"\n[JARVIS is reading your calendar]: {args['start']} to {args['end']}")
        try:
            token = calendar_token()
            params = urllib.parse.urlencode(
                {
                    "timeMin": args["start"],
                    "timeMax": args["end"],
                    "singleEvents": "true",
                    "orderBy": "startTime",
                    "maxResults": 50,
                }
            )
            req = urllib.request.Request(
                "https://www.googleapis.com/calendar/v3/calendars/primary/events?" + params,
                headers={"Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                items = json.loads(r.read().decode()).get("items", [])
            lines = []
            for e in items:
                start = e["start"].get("dateTime", e["start"].get("date"))
                end = e["end"].get("dateTime", e["end"].get("date"))
                lines.append(f"{start} -> {end} | {e.get('summary', '(no title)')}")
            return "\n".join(lines) or "No events in that range."
        except Exception as e:
            return f"Calendar error: {e}"
    return "Unknown tool."


def system_prompt():
    memory = "\n\n".join(f"## {p.stem}\n{p.read_text()}" for p in NOTES.glob("*.md"))
    now = datetime.datetime.now().astimezone().isoformat(timespec="minutes")
    return (
        "You are JARVIS, a capable personal assistant and coding partner for a "
        "programmer on a Mac. Be direct and high quality. Save durable facts about "
        "the user with save_note. For GitHub: use github_api to READ things, and use "
        "github_open_pr to create or change files (it makes the branch, commits the "
        "files and opens the pull request in one approved step; pass plain file text, "
        "never base64). For coding tasks that need tests, work on a local clone: "
        "work_clone, then work_read and work_write, then work_test, then work_open_pr. "
        "Only open the PR once tests pass, or say plainly why they fail. "
        "Never use run_shell for git clone or git push. "
        "Never push to the main branch directly. The gh CLI is NOT "
        "installed, so never use it. For calendar questions, use the calendar_events "
        "tool. Never disable SSL or certificate verification in any command. "
        "Your replies may be spoken aloud, so keep them concise and conversational "
        "unless the user asks for detail.\n\n"
        f"Current local date and time: {now}\n\n"
        f"Your memory notes:\n{memory or '(empty)'}"
    )


# =====================================================================
# Approvals: keyboard only. Nothing that spoken audio produces can say "yes".
# =====================================================================
def approve(prompt="Approve? (y/n): "):
    """Ask for approval on the KEYBOARD. The microphone is never read here."""
    if not sys.stdin.isatty():
        return False
    try:
        return input(prompt).strip().lower() == "y"
    except EOFError:
        return False


# =====================================================================
# Voice output: Kokoro (neural, offline) with the macOS `say` as fallback
# =====================================================================
VOICE_ON = os.environ.get("JARVIS_VOICE_ON", "1") == "1"
VOICE_ENGINE = os.environ.get("JARVIS_VOICE_ENGINE", "edge" if importlib.util.find_spec("edge_tts") else "gemini")  # "gemini", "kokoro" or "say"
VOICE_DIR = JARVIS_DIR / "voice"
KOKORO_MODEL = VOICE_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = VOICE_DIR / "voices-v1.0.bin"
KOKORO_VOICE = os.environ.get("JARVIS_KOKORO_VOICE", "bm_george")  # British male
try:
    KOKORO_SPEED = float(os.environ.get("JARVIS_KOKORO_SPEED", "1.0"))
except ValueError:
    KOKORO_SPEED = 1.0
VOICE_NAME = os.environ.get("JARVIS_VOICE", "")  # macOS voice for `say`, e.g. Daniel
VOICE_RATE = os.environ.get("JARVIS_VOICE_RATE", "")  # `say` words per minute

_kokoro = None
_kokoro_failed = False
_play_lock = threading.Lock()
_speech_proc = None
_speech_thread = None
_speech_stop = None


def say_available():
    return shutil.which("say") is not None


def load_kokoro():
    """Load Kokoro once. Returns the engine, or None (and remembers) if unavailable."""
    global _kokoro, _kokoro_failed, KOKORO_VOICE
    if _kokoro is not None or _kokoro_failed:
        return _kokoro
    if sd is None or np is None:
        _kokoro_failed = True
        return None
    if not (KOKORO_MODEL.exists() and KOKORO_VOICES.exists()):
        _kokoro_failed = True
        return None
    try:
        from kokoro_onnx import Kokoro

        _kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
        names = list(_kokoro.get_voices())
        if KOKORO_VOICE not in names:
            print(f"[Kokoro voice '{KOKORO_VOICE}' not found; using bm_george]")
            KOKORO_VOICE = "bm_george" if "bm_george" in names else names[0]
    except Exception as e:
        print(f"[Kokoro could not start ({e}); using the macOS voice instead]")
        _kokoro = None
        _kokoro_failed = True
    return _kokoro

def clean_for_speech(text):
    """Make a reply pleasant to hear: no code blocks, no URLs, no markdown symbols."""
    text = re.sub(r"```.*?```", " (code omitted) ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"https?://\S+", " link ", text)
    text = apply_forms(text, SPOKEN_FORMS)
    text = re.sub(r"[*_#>|]+", "", text)
    return re.sub(r"\s+", " ", text).strip()[:1500]


def split_for_tts(text, limit=300):
    """Cut text into sentence-sized pieces so speech starts quickly."""
    chunks, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        while len(s) > limit:
            cut = s.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(s[:cut])
            s = s[cut:].strip()
        if cur and len(cur) + 1 + len(s) > limit:
            chunks.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]


def stop_speaking():
    global _speech_proc, _speech_thread, _speech_stop
    with _play_lock:
        if _speech_stop is not None:
            _speech_stop.set()
        if sd is not None and _speech_thread is not None:
            try:
                sd.stop()
            except Exception:
                pass
    if _speech_proc is not None and _speech_proc.poll() is None:
        _speech_proc.terminate()
    _speech_proc = None
    _speech_thread = None
    _speech_stop = None


def is_speaking():
    if _speech_thread is not None and _speech_thread.is_alive():
        return True
    return _speech_proc is not None and _speech_proc.poll() is None


def wait_speaking(timeout=180):
    end = time.time() + timeout
    while is_speaking() and time.time() < end:
        time.sleep(0.05)


GEMINI_TTS_MODEL = os.environ.get("JARVIS_GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts")
GEMINI_VOICE = os.environ.get("JARVIS_GEMINI_VOICE", "en-in-concierge-11")
GEMINI_CHUNK = 500  # characters per Gemini request (fewer requests = friendlier to free limits)
_gemini_down_until = 0.0
EDGE_VOICE = os.environ.get("JARVIS_EDGE_VOICE", "en-IN-PrabhatNeural")
EDGE_RATE = os.environ.get("JARVIS_EDGE_RATE", "+0%")  # e.g. "+10%" faster, "-10%" slower
_edge_down_until = 0.0

# Spoken forms: used for every engine. Typed text is never changed.
SPOKEN_FORMS = dict(PERSONAL.get("spoken_forms", {}))
# Respellings for the backup voices only (Gemini says the name correctly on its own).
BACKUP_FORMS = dict(PERSONAL.get("backup_forms", {}))


def apply_forms(text, forms):
    for pattern, spoken in forms.items():
        text = re.sub(pattern, spoken, text, flags=re.I)
    return text


def gemini_tts(text):
    """Ask Gemini to speak text. Returns (int16 samples, sample rate). Raises on any failure."""
    import io
    import wave

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("no GEMINI_API_KEY")
    body = {
        "model": GEMINI_TTS_MODEL,
        "input": [{"type": "user_input", "content": [{"type": "text", "text": text}]}],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": [{"voice": GEMINI_VOICE}]},
    }
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}")
    except Exception as e:
        raise RuntimeError(type(e).__name__)
    audio = [
        c
        for s in data.get("steps", [])
        if s.get("type") == "model_output"
        for c in s.get("content", [])
        if c.get("type") == "audio"
    ]
    if not audio:
        raise RuntimeError("no audio returned")
    with wave.open(io.BytesIO(base64.b64decode(audio[-1]["data"]))) as w:
        rate, channels = w.getframerate(), w.getnchannels()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return samples, rate


def _say_fallback(text, stop=None):
    """Last-resort voice: the built-in macOS `say`."""
    global _speech_proc
    if not say_available() or (stop is not None and stop.is_set()):
        return
    cmd = ["say"]
    if VOICE_NAME:
        cmd += ["-v", VOICE_NAME]
    if VOICE_RATE.isdigit():
        cmd += ["-r", VOICE_RATE]
    try:
        _speech_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
        _speech_proc.stdin.write(apply_forms(text, BACKUP_FORMS))  # via stdin, never a shell
        _speech_proc.stdin.close()
    except OSError:
        _speech_proc = None


try:
    FIRST_CHUNK = int(os.environ.get("JARVIS_FIRST_CHUNK", "160"))  # characters; 0 = old behaviour
except ValueError:
    FIRST_CHUNK = 160


def edge_tts_file(text):
    """Synthesise text with a Microsoft Edge voice into a temporary .mp3 and return its path."""
    import asyncio
    import tempfile

    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("edge-tts is not installed")
    fd, path = tempfile.mkstemp(suffix=".mp3", prefix="jarvis_")
    os.close(fd)

    async def go():
        await asyncio.wait_for(edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE).save(path), 20)

    try:
        asyncio.run(go())
        if os.path.getsize(path) < 200:
            raise RuntimeError("no audio returned")
    except BaseException:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    return path


def _wait_proc(stop):
    """Wait for the audio player started by _play_mp3 (or `say`) to finish, unless stopped."""
    while True:
        p = _speech_proc
        if p is None or p.poll() is not None or stop.is_set():
            return
        time.sleep(0.03)


def _play_mp3(path, stop, first=False):
    """Play an mp3 with macOS afplay once the previous piece has finished. False if stopped or it failed."""
    global _speech_proc
    _wait_proc(stop)
    if stop.is_set():
        return False
    try:
        _speech_proc = subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        print(f"\n[could not play audio: {e}]")
        return False
    if first:
        tlog("reply ready -> voice starts", time.time() - _clock["t0"])
    return True


def tts_pieces(text):
    """Gemini voice: a short first piece so speech starts quickly, bigger pieces after."""
    if VOICE_ENGINE not in ("gemini", "edge"):
        return split_for_tts(text, 300)
    if FIRST_CHUNK <= 0:
        return split_for_tts(text, GEMINI_CHUNK)
    first = split_for_tts(text, FIRST_CHUNK)
    if not first or not text.startswith(first[0]):
        return split_for_tts(text, GEMINI_CHUNK)
    rest = text[len(first[0]):].strip()
    return [first[0]] + (split_for_tts(rest, GEMINI_CHUNK) if rest else [])


def _tts_worker(text, stop):
    """Speak text sentence-group by sentence-group: Gemini first, then Kokoro, then `say`."""
    global _gemini_down_until, _edge_down_until
    pieces = tts_pieces(text)
    edge_files = []
    use_edge = VOICE_ENGINE == "edge" and time.time() >= _edge_down_until
    use_gemini = VOICE_ENGINE == "gemini" and time.time() >= _gemini_down_until
    queued = False
    try:
        for i, piece in enumerate(pieces):
            if stop.is_set():
                return
            samples = rate = None
            if use_edge:
                path = None
                try:
                    path = edge_tts_file(piece)
                    edge_files.append(path)
                except Exception as e:
                    print(f"\n[Edge voice unavailable ({e}); using the backup voice for a while]")
                    _edge_down_until = time.time() + 120
                    use_edge = False
                if path:
                    if not _play_mp3(path, stop, not queued):
                        return
                    queued = True
                    continue
            if use_gemini:
                try:
                    samples, rate = gemini_tts(piece)
                except Exception as e:
                    print(f"\n[Gemini voice unavailable ({e}); using the backup voice for a while]")
                    _gemini_down_until = time.time() + 120
                    use_gemini = False
            if samples is None:
                kokoro = load_kokoro()
                if kokoro is None:
                    if queued:
                        sd.wait()
                    _say_fallback(" ".join(pieces[i:]), stop)
                    return
                lang = "en-gb" if KOKORO_VOICE.startswith("b") else "en-us"
                samples, rate = kokoro.create(
                    apply_forms(piece, BACKUP_FORMS), voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang=lang
                )
            if queued:
                sd.wait()  # let the previous piece finish while this one was being made
            with _play_lock:
                if stop.is_set():
                    return
                sd.play(samples, rate)
            if not queued:
                tlog("reply ready -> voice starts", time.time() - _clock["t0"])
            queued = True
        if queued and not stop.is_set():
            _wait_proc(stop)
            if sd is not None:
                sd.wait()
    except Exception as e:
        print(f"\n[voice error: {e}]")
    finally:
        for f in edge_files:
            try:
                os.remove(f)
            except OSError:
                pass


def speak(text):
    """Speak text without blocking. Gemini voice, then Kokoro, then macOS `say` as fallbacks."""
    global _speech_thread, _speech_stop
    stop_speaking()
    if not VOICE_ON:
        return
    text = clean_for_speech(text)
    if not text:
        return
    if VOICE_ENGINE == "edge" or (sd is not None and np is not None and VOICE_ENGINE in ("gemini", "kokoro")):
        _clock["t0"] = time.time()
        _speech_stop = threading.Event()
        _speech_thread = threading.Thread(target=_tts_worker, args=(text, _speech_stop), daemon=True)
        _speech_thread.start()
    else:
        _say_fallback(text)


def voice_label():
    if not VOICE_ON:
        return "off"
    if VOICE_ENGINE == "edge":
        return f"Edge ({EDGE_VOICE}), backup Kokoro/say"
    if VOICE_ENGINE == "gemini":
        return f"Gemini ({GEMINI_VOICE}), backup Kokoro/say"
    if VOICE_ENGINE == "kokoro" and _kokoro is not None:
        return f"Kokoro ({KOKORO_VOICE})"
    if say_available():
        return f"macOS say ({VOICE_NAME or 'default voice'})"
    return "off (no voice engine found)"


def installed_say_voices():
    """Return [(name, locale)] from `say -v ?`."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    voices = []
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s+([a-z]{2,3}_[A-Z0-9]+)\s+#", line)
        if m:
            voices.append((m.group(1).strip(), m.group(2)))
    return voices


def handle_voice_command(cmd):
    global VOICE_ON, VOICE_ENGINE, VOICE_NAME, KOKORO_VOICE, GEMINI_VOICE, EDGE_VOICE, _edge_down_until
    head = cmd.split(None, 1)
    rest = head[1].strip() if len(head) > 1 else ""
    low = rest.lower()
    word, _, tail = rest.partition(" ")
    if head[0] == "/voices":
        print("Edge voices: run  edge-tts --list-voices | grep en-IN  then  /voice edge en-IN-PrabhatNeural")
        print("Gemini voices: run  python gemini_tts_test.py voices en-IN  (or en-US) in a new Terminal tab,")
        print("  then use  /voice gemini VOICE_ID")
        kokoro = load_kokoro()
        if kokoro is not None:
            print("Kokoro voices: " + ", ".join(n for n in kokoro.get_voices() if n[:2] in ("af", "am", "bf", "bm")))
        say_names = [n for n, loc in installed_say_voices() if loc.startswith("en_")]
        if say_names:
            print("macOS voices: " + ", ".join(say_names))
        print()
    elif low == "off":
        VOICE_ON = False
        stop_speaking()
        print("Voice off.\n")
    elif low == "on":
        VOICE_ON = True
        print(f"Voice on: {voice_label()}.\n")
    elif word.lower() == "edge":
        if importlib.util.find_spec("edge_tts") is None:
            print("Edge voices need:  pip install edge-tts\n")
            return
        if tail.strip():
            EDGE_VOICE = tail.strip()
        VOICE_ENGINE = "edge"
        VOICE_ON = True
        _edge_down_until = 0.0
        speak(f"Hello{NAME_SUFFIX}, I am JARVIS. How do I sound?")
        print(f"Voice: {voice_label()}.\n")
    elif word.lower() == "gemini":
        if tail.strip():
            GEMINI_VOICE = tail.strip()
        VOICE_ENGINE = "gemini"
        VOICE_ON = True
        speak(f"Hello{NAME_SUFFIX}, I am JARVIS. How do I sound?")
        print(f"Voice: {voice_label()}.\n")
    elif low in ("say", "kokoro"):
        if low == "kokoro" and load_kokoro() is None:
            print("Kokoro is not available. Check the install steps.\n")
            return
        VOICE_ENGINE = low
        VOICE_ON = True
        speak("Hello, I am JARVIS. How do I sound?")
        print(f"Voice: {voice_label()}.\n")
    elif rest:
        kokoro = load_kokoro()
        kokoro_names = list(kokoro.get_voices()) if kokoro is not None else []
        if low in [n.lower() for n in kokoro_names]:
            KOKORO_VOICE = [n for n in kokoro_names if n.lower() == low][0]
            VOICE_ENGINE = "kokoro"
        else:
            match = [n for n, _ in installed_say_voices() if n.lower() == low]
            if not match:
                print(f"No voice called '{rest}'. Type /voices to see the list.\n")
                return
            VOICE_NAME = match[0]
            VOICE_ENGINE = "say"
        VOICE_ON = True
        speak("Hello, I am JARVIS. This is my voice.")
        print(f"Voice: {voice_label()}.\n")
    else:
        print(f"Voice: {voice_label()}. Use /voice on, /voice off, /voice edge [VOICE], /voice gemini [VOICE_ID], "
              "/voice kokoro, /voice say, /voices, or /voice NAME.\n")



# =====================================================================
# Voice input: /talk (type /talk, then speak) and /listen ("Hey Jarvis")
# =====================================================================
RATE = 16000
BLOCK = 1280  # 80 ms of audio per block
BLOCK_SECS = BLOCK / RATE
WHISPER_MODEL = os.environ.get("JARVIS_WHISPER_MODEL", "base.en")
END_PHRASE = os.environ.get("JARVIS_END_PHRASE", "send it")
CANCEL_PHRASE = os.environ.get("JARVIS_CANCEL_PHRASE", "cancel that")
SLEEP_PHRASE = os.environ.get("JARVIS_SLEEP_PHRASE", "go to sleep")
try:
    FOLLOWUP = float(os.environ.get("JARVIS_FOLLOWUP", "30"))  # seconds of follow-up listening; 0 = off
except ValueError:
    FOLLOWUP = 30.0

_listen_stop = threading.Event()  # set by the web app (or anything else) to leave hands-free mode
LISTEN_HOOK = None  # optional callable(status) used by the web app
LISTEN_STATE = {"status": "off"}


class ListenStopped(Exception):
    pass


def listen_status(status):
    """Record (and report) what hands-free mode is doing: wake, listening, thinking, speaking, off."""
    LISTEN_STATE["status"] = status
    if LISTEN_HOOK:
        try:
            LISTEN_HOOK(status)
        except Exception:
            pass
try:
    WAKE_THRESHOLD = float(os.environ.get("JARVIS_WAKE_THRESHOLD", "0.5"))
except ValueError:
    WAKE_THRESHOLD = 0.5
WAKE_DEBUG = os.environ.get("JARVIS_WAKE_DEBUG", "0") == "1"
MIC_THRESHOLD = 300.0  # replaced by calibrate(); override with JARVIS_MIC_THRESHOLD
try:
    SILENCE_END = min(max(float(os.environ.get("JARVIS_SILENCE", "0.6")), 0.3), 1.5)  # quiet that ends a chunk
except ValueError:
    SILENCE_END = 0.6
_whisper = None
_wakeword = None


def ensure_audio(wake=False):
    """Return an error message if a needed package is missing, else None."""
    missing = []
    if sd is None:
        missing.append("sounddevice")
    if np is None:
        missing.append("numpy")
    if importlib.util.find_spec("faster_whisper") is None:
        missing.append("faster-whisper")
    if wake and importlib.util.find_spec("openwakeword") is None:
        missing.append("openwakeword")
    if missing:
        return ("Missing packages: " + ", ".join(missing) + ".\nInstall with:\n  pip install "
                + " ".join(missing))
    return None


class Mic:
    """Reads the microphone into a queue so no audio is lost while Whisper is busy.
    The microphone is only open inside a `with Mic()` block."""

    def __enter__(self):
        self.q = queue.Queue()
        self.stream = sd.InputStream(
            samplerate=RATE, channels=1, dtype="int16", blocksize=BLOCK, callback=self._cb
        )
        self.stream.start()
        return self

    def _cb(self, indata, frames, t, status):
        self.q.put(indata[:, 0].copy())

    def read(self, timeout=3.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError("No audio is coming from the microphone.")

    def __exit__(self, *exc):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


def rms(block):
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def calibrate(mic):
    """Measure room noise for ~0.7 s and set the speech threshold."""
    global MIC_THRESHOLD
    override = os.environ.get("JARVIS_MIC_THRESHOLD")
    if override:
        try:
            MIC_THRESHOLD = float(override)
            return
        except ValueError:
            pass
    levels = [rms(mic.read()) for _ in range(9)]
    if max(levels) == 0.0:
        raise RuntimeError(
            "The microphone is returning silence. Allow Terminal under System Settings > "
            "Privacy & Security > Microphone, then restart Terminal."
        )
    noise = sorted(levels)[len(levels) // 2]
    MIC_THRESHOLD = min(max(250.0, noise * 3.0), 2500.0)


_whisper_lock = threading.Lock()


def load_whisper(quiet=False):
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel

            if not quiet:
                print(f"[Loading speech recognition '{WHISPER_MODEL}'. The first run downloads it.]")
            t0 = time.time()
            _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            if not quiet:
                tlog("speech recognition loaded", time.time() - t0)
    return _whisper


def preload_whisper():
    """Load speech recognition in the background at startup so the first /talk or /listen is quick."""
    if os.environ.get("JARVIS_PRELOAD", "1") != "1":
        return
    try:
        if importlib.util.find_spec("faster_whisper") is None:
            return
    except (ValueError, ImportError):
        return

    def _go():
        try:
            load_whisper(quiet=True)
        except Exception:
            pass  # a normal load later will show the real error

    threading.Thread(target=_go, daemon=True).start()


def load_wakeword():
    global _wakeword
    if _wakeword is None:
        from openwakeword.model import Model

        try:
            _wakeword = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        except Exception:
            import openwakeword

            print("[Downloading the 'hey jarvis' wake-word model from GitHub...]")
            openwakeword.utils.download_models(["hey_jarvis"])
            _wakeword = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    return _wakeword


def transcribe(audio_int16):
    audio = audio_int16.astype(np.float32) / 32768.0
    model = load_whisper()
    t0 = time.time()
    segments, _ = model.transcribe(
        audio, language="en", beam_size=1, vad_filter=True, condition_on_previous_text=False
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    tlog(f"speech-to-text, {len(audio) / RATE:.0f}s of audio", time.time() - t0)
    if norm(text) in ("you", "thanks for watching", "thank you for watching"):
        return ""  # typical Whisper hallucinations on near-silence
    return text


def norm(text):
    """Lowercase, 'I'm'/'Im' -> 'i am', drop punctuation, so phrase matching is forgiving."""
    t = text.lower().replace("\u2019", "'")
    t = re.sub(r"\bi'?m\b", "i am", t)
    t = t.replace("'", "")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def strip_phrase(text, phrase):
    """If `text` ENDS with `phrase` (at a word boundary), return text without it, else None.
    A phrase in the middle of a sentence never matches."""
    target = norm(phrase)
    if not target:
        return None
    tokens = text.split()
    for k in range(1, len(tokens) + 1):
        if norm(" ".join(tokens[-k:])) == target:
            return " ".join(tokens[:-k]).rstrip(" ,;:-")
    return None


def record_chunk(mic, idle_limit, silence_end=None, max_secs=30.0):
    """Wait for speech, record until a pause. Returns int16 audio, or None if nothing was
    said for idle_limit seconds."""
    silence_end = SILENCE_END if silence_end is None else silence_end
    pre = collections.deque(maxlen=4)  # ~0.3 s of lead-in so first words aren't clipped
    frames, speaking, voiced, silent, waited = [], False, 0, 0.0, 0.0
    while True:
        if _listen_stop.is_set():
            raise ListenStopped()
        block = mic.read()
        loud = rms(block) > MIC_THRESHOLD
        if not speaking:
            waited += BLOCK_SECS
            if loud:
                speaking, frames, voiced, silent = True, list(pre) + [block], 1, 0.0
            else:
                pre.append(block)
                if waited >= idle_limit:
                    return None
            continue
        frames.append(block)
        if loud:
            voiced += 1
            silent = 0.0
        else:
            silent += BLOCK_SECS
        if silent >= silence_end or len(frames) * BLOCK_SECS >= max_secs:
            if voiced >= 3:  # ignore clicks and bumps
                return np.concatenate(frames)
            speaking, frames, voiced, silent = False, [], 0, 0.0


def capture_message(mic, first_idle=10.0):
    """Listen, one pause-separated chunk at a time, until the end or cancel phrase.
    Returns ("send", text), ("cancel", ""), ("sleep", "") or ("timeout", "")."""
    parts = []
    while True:
        chunk = record_chunk(mic, first_idle if not parts else 60.0)
        if chunk is None:
            if parts:
                print("  Timed out waiting for the end phrase. Message discarded.")
            return ("timeout", "")
        text = transcribe(chunk)
        if not text:
            continue
        print(f"  [heard] {text}")
        if strip_phrase(text, CANCEL_PHRASE) is not None:
            return ("cancel", "")
        if not parts and strip_phrase(text, SLEEP_PHRASE) == "":
            return ("sleep", "")
        rest = strip_phrase(text, END_PHRASE)
        if rest is not None:
            if rest:
                parts.append(rest)
            final = " ".join(parts).strip()
            if not final:
                print("  (nothing to send yet)")
                continue
            return ("send", final)
        parts.append(text)


def chime(name):
    """Short system sound (Glass = listening, Pop = sent, Basso = cancelled)."""
    if os.environ.get("JARVIS_CHIME", "1") != "1" or not shutil.which("afplay"):
        return
    try:
        subprocess.Popen(  # fire and forget, so listening starts immediately
            ["afplay", f"/System/Library/Sounds/{name}.aiff"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def talk_once():
    """/talk: speak a message, finish with the end phrase. Returns the text or None."""
    err = ensure_audio()
    if err:
        print(err + "\n")
        return None
    stop_speaking()
    load_whisper()
    with Mic() as mic:
        print("Calibrating... stay quiet for a second.")
        calibrate(mic)
        print(f"Listening. Say \"{END_PHRASE}\" to send or \"{CANCEL_PHRASE}\" to discard. "
              "Ctrl+C stops.")
        status, text = capture_message(mic)
    if status == "cancel":
        print("  Cancelled.\n")
    elif status == "timeout":
        print("  Heard nothing, so I stopped listening.\n")
    return text if status == "send" else None


def wait_for_wakeword(model, mic):
    """Block until 'Hey Jarvis' is heard on the (already open) microphone. Nothing is transcribed or saved."""
    if hasattr(model, "reset"):
        model.reset()
    while True:
        if _listen_stop.is_set():
            raise ListenStopped()
        scores = model.predict(mic.read())
        top = max(scores.values()) if scores else 0.0
        if WAKE_DEBUG and top > 0.2:
            print(f"  [wake score {top:.2f}]")
        if top >= WAKE_THRESHOLD:
            return


def listen_loop(messages):
    """/listen: hands-free. 'Hey Jarvis' once, then keep talking: speak -> end phrase -> reply -> speak again.
    Say the sleep phrase (or stay quiet for JARVIS_FOLLOWUP seconds) to go back to waiting for 'Hey Jarvis'."""
    err = ensure_audio(wake=True)
    if err:
        print(err + "\n")
        return
    stop_speaking()
    _listen_stop.clear()
    try:
        load_whisper()
        model = load_wakeword()
        with Mic() as mic:
            print("Calibrating... stay quiet for a second.")
            calibrate(mic)
        if not VOICE_ON:
            print("Note: voice replies are off, so you will not hear answers (/voice on).")
        print(f"Hands-free mode. Say \"Hey Jarvis\" once, then your message, then \"{END_PHRASE}\" "
              f"(or \"{CANCEL_PHRASE}\"). After a reply I keep listening for {FOLLOWUP:.0f}s; "
              f"say \"{SLEEP_PHRASE}\" to stop. Approvals need the keyboard/web buttons. Ctrl+C to leave.\n")
        in_convo = False
        while True:
            with Mic() as mic:
                if not in_convo:
                    listen_status("wake")
                    wait_for_wakeword(model, mic)
                    chime("Glass")
                    print("[Listening...]")
                else:
                    print(f"[Still listening for {FOLLOWUP:.0f}s...]")
                listen_status("listening")
                status, text = capture_message(mic, FOLLOWUP if in_convo else 10.0)
            if status == "send":
                chime("Pop")
                print(f"\nYou (voice): {text}")
                listen_status("thinking")
                run_turn(messages, text)
                listen_status("speaking")
                wait_speaking()  # microphone stays closed while JARVIS talks...
                time.sleep(0.4)  # ...and a moment afterwards
                in_convo = FOLLOWUP > 0
            elif status == "cancel":
                chime("Basso")
                print("  Cancelled.\n")
            elif status == "sleep":
                print("  Going back to sleep. Say \"Hey Jarvis\" to wake me.\n")
                in_convo = False
            else:
                print("  Heard nothing. Waiting for \"Hey Jarvis\" again.\n")
                in_convo = False
    except (KeyboardInterrupt, ListenStopped):
        stop_speaking()
        print("\nLeft hands-free mode.\n")
    except Exception as e:
        stop_speaking()
        print(f"\n[Hands-free mode stopped: {e}]\n")
    finally:
        listen_status("off")


# =====================================================================
# Conversation
# =====================================================================
HELP = """Commands:
  /talk            speak one message; finish with your end phrase
  /listen          hands-free: say "Hey Jarvis" once, then just keep talking (Ctrl+C to leave)
                   say "go to sleep" to make it wait for "Hey Jarvis" again
  /phrases         show the end and cancel phrases
  /endphrase X     change the end phrase for this session
  /cancelphrase X  change the cancel phrase for this session
  /voice on|off|edge [VOICE]|gemini [ID]|kokoro|say|NAME    /voices    list voices
  /help            this list
Set JARVIS_END_PHRASE / JARVIS_CANCEL_PHRASE in ~/.zshrc to make phrases permanent.
"""


def run_turn(messages, user_text):
    """Send one user message and run the tool loop until JARVIS gives a final answer."""
    mark = len(messages)
    messages.append({"role": "user", "content": user_text})
    try:
        while True:
            t0 = time.time()
            resp = chat_call(messages)
            tlog("model reply", time.time() - t0)
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            if msg.content:
                print(f"\nJARVIS: {msg.content}\n")
            if not msg.tool_calls:
                if msg.content:
                    speak(msg.content)
                return
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": run_tool(tc.function.name, args),
                    }
                )
    except KeyboardInterrupt:
        del messages[mark:]
        print("\n[Interrupted]\n")
        raise
    except Exception as e:
        del messages[mark:]
        print(f"\n[Error talking to the model: {e}]\n")


def main():
    global END_PHRASE, CANCEL_PHRASE
    messages = []
    print(f"JARVIS online ({PROVIDER}: {MODEL}). Ctrl+C to exit.")
    if VOICE_ON and VOICE_ENGINE == "kokoro":
        load_kokoro()
    print(f"Voice: {voice_label()}  (/help for commands; /talk and /listen for voice input)\n")
    preload_whisper()
    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            stop_speaking()
            print("\nGoodbye.")
            break
        stop_speaking()
        if not user:
            continue
        low = user.lower()
        if low == "/help":
            print(HELP)
            continue
        if low.startswith("/voice"):
            handle_voice_command(user)
            continue
        if low == "/phrases":
            print(f"End phrase: \"{END_PHRASE}\"   Cancel phrase: \"{CANCEL_PHRASE}\"\n")
            continue
        if low.startswith(("/endphrase", "/cancelphrase")):
            parts = user.split(None, 1)
            new = parts[1].strip() if len(parts) > 1 else ""
            if not new or len(new.split()) > 6:
                print("Give a phrase of 1 to 6 words, e.g. /endphrase over to you\n")
            elif low.startswith("/endphrase"):
                END_PHRASE = new
                print(f"End phrase is now \"{END_PHRASE}\".\n")
            else:
                CANCEL_PHRASE = new
                print(f"Cancel phrase is now \"{CANCEL_PHRASE}\".\n")
            continue
        if low == "/talk":
            try:
                text = talk_once()
            except KeyboardInterrupt:
                print("\nStopped listening.\n")
                continue
            except Exception as e:
                print(f"[Voice input error: {e}]\n")
                continue
            if not text:
                continue
            print(f"\nYou (voice): {text}")
            user = text
        elif low == "/listen":
            listen_loop(messages)
            continue
        try:
            run_turn(messages, user)
        except KeyboardInterrupt:
            continue


if __name__ == "__main__":
    main()