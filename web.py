"""JARVIS web app: a private control panel for jarvis.py, served only on THIS Mac.

  python3 web.py            then open the private link it prints (first time only; the browser remembers)

Privacy / safety:
  * Listens on 127.0.0.1 only, so nothing else on your network (or the internet) can reach it.
  * Needs a secret token (saved in ~/jarvis/web_token, readable only by you). Without the token
    every request gets "401". The token is exchanged for a cookie, then removed from the address bar.
  * Rejects requests whose Host or Origin is not this machine (blocks DNS-rebinding and other websites).
  * Approvals for shell commands / GitHub / tests are buttons on the page. The model and the microphone
    can never approve anything.
Uses the same jarvis.py: same notes, same tools, same voice. Needs no extra packages.
"""
import hmac
import json
import os
import pathlib
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import jarvis as jv  # noqa: E402  (the normal jarvis.py, imported as a module)

HOST = "127.0.0.1"
try:
    PORT = int(os.environ.get("JARVIS_WEB_PORT", "8765"))
except ValueError:
    PORT = 8765
COOKIE = "jarvis_session"

# ---------------------------------------------------------------- token
TOKEN_FILE = jv.JARVIS_DIR / "web_token"
jv.JARVIS_DIR.mkdir(parents=True, exist_ok=True)
if TOKEN_FILE.exists() and TOKEN_FILE.read_text().strip():
    TOKEN = TOKEN_FILE.read_text().strip()
else:
    TOKEN = secrets.token_urlsafe(32)
    TOKEN_FILE.write_text(TOKEN)
try:
    os.chmod(TOKEN_FILE, 0o600)
except OSError:
    pass

# ---------------------------------------------------------------- shared state
_lock = threading.RLock()
_events = []          # chat bubbles, system notes and activity log lines
_next_id = [1]
messages = []         # the conversation, exactly like the terminal version
turn_lock = threading.Lock()
_state = {"busy": False, "mic": False}
_approvals = {}       # id -> {"text", "event", "ok"}
_cap = threading.local()
_recent = []          # recent raw output, used to show what an approval is about


def emit(kind, text):
    with _lock:
        ev = {"id": _next_id[0], "kind": kind, "text": text, "t": time.strftime("%H:%M:%S")}
        _next_id[0] += 1
        _events.append(ev)
        if len(_events) > 3000:
            del _events[:500]
    return ev


class Tee:
    """Copies everything jarvis.py prints into the page's Activity log (and still prints in Terminal)."""

    def __init__(self, real):
        self.real = real
        self.buf = ""

    def write(self, s):
        try:
            self.real.write(s)
        except Exception:
            pass
        cap = getattr(_cap, "buf", None)
        if cap is not None:
            cap.append(s)
        with _lock:
            _recent.append(s)
            del _recent[:-200]
            self.buf += s
            while "\n" in self.buf:
                line, self.buf = self.buf.split("\n", 1)
                if line.strip():
                    emit("log", line.rstrip())
        return len(s)

    def flush(self):
        try:
            self.real.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.real, name)


# ---------------------------------------------------------------- approvals (buttons only)
def web_approve(prompt="Approve? (y/n): "):
    with _lock:
        text = "".join(_recent)
    i = text.rfind("[JARVIS wants")
    detail = text[i:] if i >= 0 else "JARVIS is asking for approval."
    aid = secrets.token_hex(4)
    item = {"text": detail.strip()[-3000:], "event": threading.Event(), "ok": False}
    with _lock:
        _approvals[aid] = item
    emit("system", "Waiting for your approval (see the box above the message field).")
    item["event"].wait(timeout=600)
    with _lock:
        _approvals.pop(aid, None)
    if not item["event"].is_set():
        emit("system", "Approval timed out, so it was denied.")
        return False
    return item["ok"]


jv.approve = web_approve  # run_tool() calls approve() by name, so this replaces the keyboard prompt


# ---------------------------------------------------------------- turns
_orig_run_turn = jv.run_turn


def run_turn_web(msgs, text):
    """Same as the terminal run_turn, plus chat bubbles. Also used by hands-free mode."""
    with turn_lock:
        _state["busy"] = True
        who = threading.current_thread().name
        emit("user", ("\U0001F399 " if who.startswith("jarvis-") else "") + text)
        mark = len(msgs)
        try:
            _orig_run_turn(msgs, text)
        finally:
            _state["busy"] = False
        got = False
        for m in msgs[mark:]:
            if m.get("role") == "assistant" and m.get("content"):
                emit("assistant", m["content"])
                got = True
        if not got:
            emit("system", "No reply this time. Open Activity to see what went wrong.")


jv.run_turn = run_turn_web


def run_in_thread(name, fn, *args):
    def target():
        _cap.buf = None
        try:
            fn(*args)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            emit("system", f"{name} stopped: {e}")
    t = threading.Thread(target=target, name=name, daemon=True)
    t.start()
    return t


_listen_thread = [None]
_talk_thread = [None]


def listening():
    t = _listen_thread[0]
    return bool(t and t.is_alive())


def talking():
    t = _talk_thread[0]
    return bool(t and t.is_alive())


def start_listen():
    if listening() or talking():
        return "Already listening."
    _listen_thread[0] = run_in_thread("jarvis-listen", jv.listen_loop, messages)
    return "Hands-free is starting. Say \"Hey Jarvis\" once, then keep talking."


def stop_listen():
    jv._listen_stop.set()
    jv.stop_speaking()


def talk_job():
    jv._listen_stop.clear()
    text = jv.talk_once()
    if text:
        run_turn_web(messages, text)


def start_talk():
    if listening() or talking():
        return "The microphone is already in use."
    _talk_thread[0] = run_in_thread("jarvis-talk", talk_job)
    return "Listening for one message. Say \"" + jv.END_PHRASE + "\" to send it."


def command(text):
    """Slash commands, same as the terminal. Returns text to show, or None."""
    low = text.lower()
    parts = text.split(None, 1)
    if low == "/help":
        return jv.HELP
    if low.startswith("/voice"):
        _cap.buf = []
        try:
            jv.handle_voice_command(text)
        finally:
            out = "".join(_cap.buf)
            _cap.buf = None
        return out.strip() or "Done."
    if low == "/phrases":
        return f'End phrase: "{jv.END_PHRASE}"   Cancel phrase: "{jv.CANCEL_PHRASE}"   Sleep phrase: "{jv.SLEEP_PHRASE}"'
    if low.startswith(("/endphrase", "/cancelphrase")):
        new = parts[1].strip() if len(parts) > 1 else ""
        if not new or len(new.split()) > 6:
            return "Give a phrase of 1 to 6 words, e.g. /endphrase over to you"
        if low.startswith("/endphrase"):
            jv.END_PHRASE = new
            return f'End phrase is now "{new}".'
        jv.CANCEL_PHRASE = new
        return f'Cancel phrase is now "{new}".'
    if low == "/talk":
        return start_talk()
    if low == "/listen":
        return start_listen()
    if low == "/clear":
        messages.clear()
        return "Conversation cleared (your saved notes are untouched)."
    return "Unknown command. Type /help."


def handle_send(text):
    text = text.strip()
    if not text:
        return 400, {"error": "empty"}
    jv.stop_speaking()
    if text.startswith("/"):
        emit("user", text)
        out = command(text)
        if out:
            emit("system", out)
        return 200, {"ok": True}
    if not turn_lock.acquire(blocking=False):
        return 409, {"error": "JARVIS is still working on the last message."}
    turn_lock.release()
    run_in_thread("web-turn", run_turn_web, messages, text)
    return 200, {"ok": True}


def snapshot(after):
    with _lock:
        evs = [e for e in _events if e["id"] > after]
        appr = [{"id": k, "text": v["text"]} for k, v in _approvals.items()]
        last = _next_id[0] - 1
    return {
        "events": evs[-400:], "last": last, "busy": _state["busy"],
        "listen": jv.LISTEN_STATE["status"] if listening() else ("mic" if talking() else "off"),
        "voice": jv.voice_label(), "voice_on": bool(jv.VOICE_ON),
        "model": f"{jv.PROVIDER}: {jv.MODEL}", "approvals": appr,
        "end_phrase": jv.END_PHRASE, "sleep_phrase": jv.SLEEP_PHRASE,
    }


# ---------------------------------------------------------------- HTTP
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}


class Handler(BaseHTTPRequestHandler):
    server_version = "JarvisWeb"

    def log_message(self, *a):
        pass

    # -- helpers
    def _send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text") or ctype == "application/json" else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                         "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _host_ok(self):
        if self.headers.get("Host", "") not in ALLOWED_HOSTS:
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"}:
            return False
        return True

    def _authed(self):
        for part in self.headers.get("Cookie", "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE and hmac.compare_digest(v.encode(), TOKEN.encode()):
                return True
        return False

    # -- routes
    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {"error": "forbidden"})
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/" and "token" in q:
            if hmac.compare_digest(q["token"][0].encode(), TOKEN.encode()):
                return self._send(302, b"", "text/plain", {
                    "Location": "/",
                    "Set-Cookie": f"{COOKIE}={TOKEN}; HttpOnly; SameSite=Strict; Path=/; Max-Age=31536000",
                })
            return self._send(401, "Wrong link. Use the link printed in Terminal.", "text/plain")
        if not self._authed():
            return self._send(401, "Private. Open the link that web.py printed in Terminal.", "text/plain")
        if u.path == "/":
            return self._send(200, PAGE, "text/html")
        if u.path == "/api/state":
            try:
                after = int(q.get("after", ["0"])[0])
            except ValueError:
                after = 0
            return self._send(200, snapshot(after))
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._host_ok():
            return self._send(403, {"error": "forbidden"})
        if not self._authed() or self.headers.get("X-Jarvis") != "1":
            return self._send(401, {"error": "unauthorized"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 65536:
                return self._send(413, {"error": "too large"})
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "bad json"})
        path = urlparse(self.path).path
        if path == "/api/send":
            code, out = handle_send(str(body.get("text", "")))
            return self._send(code, out)
        if path == "/api/approve":
            with _lock:
                item = _approvals.get(str(body.get("id", "")))
            if not item:
                return self._send(404, {"error": "nothing to approve"})
            item["ok"] = bool(body.get("ok"))
            item["event"].set()
            emit("system", "Approved." if item["ok"] else "Denied.")
            return self._send(200, {"ok": True})
        if path == "/api/listen":
            if body.get("on"):
                emit("system", start_listen())
            else:
                stop_listen()
            return self._send(200, {"ok": True})
        if path == "/api/talk":
            emit("system", start_talk())
            return self._send(200, {"ok": True})
        if path == "/api/stopmic":
            stop_listen()
            return self._send(200, {"ok": True})
        if path == "/api/stopvoice":
            jv.stop_speaking()
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})


def main():
    sys.stdout = Tee(sys.stdout)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    link = f"http://127.0.0.1:{PORT}/?token={TOKEN}"
    real = sys.stdout.real
    real.write(f"\nJARVIS web app is running (only on this Mac).\nOpen this private link once:\n\n  {link}\n\n"
               "Afterwards http://127.0.0.1:%d works in that browser. Ctrl+C here to stop.\n\n" % PORT)
    real.flush()
    if jv.VOICE_ON and jv.VOICE_ENGINE == "kokoro":
        jv.load_kokoro()
    jv.preload_whisper()
    if os.environ.get("JARVIS_WEB_OPEN", "1") == "1":
        threading.Timer(0.8, lambda: webbrowser.open(link)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        jv._listen_stop.set()
        jv.stop_speaking()
        real.write("\nStopped.\n")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JARVIS</title>
<style>
:root{--bg:#0b0f14;--panel:#121923;--line:#1f2b3a;--text:#e6edf5;--dim:#8a9bb0;--accent:#4cc9f0;--ok:#3ddc97;--warn:#ffb454;--bad:#ff6b6b;--me:#17324a;--onacc:#04202b}
@media (prefers-color-scheme: light){:root{--bg:#f4f7fa;--panel:#fff;--line:#d9e2ec;--text:#14202e;--dim:#5b6b7e;--accent:#0a7aa8;--ok:#138a5b;--warn:#b26a00;--bad:#c0392b;--me:#dff1fb;--onacc:#fff}}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:10px;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
h1{font-size:16px;letter-spacing:.18em;margin:0 8px 0 0;color:var(--accent)}
.pill{font-size:12px;padding:3px 10px;border-radius:99px;border:1px solid var(--line);color:var(--dim)}
.pill.on{color:var(--ok);border-color:var(--ok)}.pill.busy{color:var(--warn);border-color:var(--warn)}
.sp{flex:1}
button{font:inherit;font-size:13px;color:var(--text);background:transparent;border:1px solid var(--line);border-radius:8px;padding:6px 11px;cursor:pointer}
button:hover{border-color:var(--accent)}button.on{border-color:var(--ok);color:var(--ok)}
button.primary{background:var(--accent);border-color:var(--accent);color:var(--onacc);font-weight:600}
button.danger{border-color:var(--bad);color:var(--bad)}
main{flex:1;overflow:auto;padding:18px 16px;display:flex;flex-direction:column;gap:12px;max-width:900px;width:100%;margin:0 auto}
.msg{padding:10px 14px;border-radius:12px;max-width:85%;white-space:pre-wrap;word-wrap:break-word;border:1px solid var(--line)}
.msg.user{align-self:flex-end;background:var(--me)}.msg.assistant{align-self:flex-start;background:var(--panel)}
.msg.system{align-self:center;color:var(--dim);font-size:13px;border-style:dashed;background:transparent}
.msg pre{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;margin:8px 0;white-space:pre}
code{font-family:ui-monospace,Menlo,monospace;font-size:13px}.msg :not(pre)>code{background:var(--bg);padding:1px 5px;border-radius:5px}
.empty{margin:auto;color:var(--dim);text-align:center}
#approvals{max-width:900px;width:100%;margin:0 auto;padding:0 16px}
.appr{border:1px solid var(--warn);border-radius:12px;padding:12px;margin-bottom:8px;background:var(--panel)}
.appr pre{margin:6px 0 10px;white-space:pre-wrap;word-break:break-word;font-size:13px;max-height:200px;overflow:auto}
.appr b{color:var(--warn)}.appr button{margin-right:8px}
footer{border-top:1px solid var(--line);background:var(--panel);padding:10px 16px}
.row{max-width:900px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
textarea{flex:1;resize:none;font:inherit;color:var(--text);background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:10px 12px;max-height:160px;outline:none}
textarea:focus{border-color:var(--accent)}
details{max-width:900px;margin:8px auto 0;color:var(--dim);font-size:12px}
#log{max-height:180px;overflow:auto;white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px;margin-top:6px}
.hint{max-width:900px;margin:6px auto 0;color:var(--dim);font-size:12px}
</style></head><body>
<header>
  <h1>JARVIS</h1>
  <span class="pill" id="model">...</span>
  <span class="pill" id="status">idle</span>
  <span class="sp"></span>
  <button id="hf" title="Say Hey Jarvis once, then just keep talking">Hands-free</button>
  <button id="mic" title="Speak one message, finish with your end phrase">Mic</button>
  <button id="stopv" title="Stop JARVIS talking">Stop voice</button>
  <button id="voice" title="Spoken replies on/off">Voice</button>
  <button id="clear" title="Start a fresh conversation (notes are kept)">Clear</button>
</header>
<main id="chat"><div class="empty" id="empty">Ask anything, or press Hands-free and say &ldquo;Hey Jarvis&rdquo;.</div></main>
<div id="approvals"></div>
<footer>
  <div class="row"><textarea id="box" rows="1" placeholder="Message JARVIS   (Enter to send, Shift+Enter for a new line, /help for commands)"></textarea>
  <button class="primary" id="send">Send</button></div>
  <div class="hint" id="hint"></div>
  <details><summary>Activity</summary><div id="log"></div></details>
</footer>
<script>
const $=id=>document.getElementById(id);
let last=0, busyShown=false, state={};
async function post(path,body){
  const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json","X-Jarvis":"1"},body:JSON.stringify(body||{})});
  if(r.status===401){document.body.innerHTML="<p style='padding:30px'>Session expired. Open the private link from Terminal again.</p>";}
  return r;
}
function rich(el,text){
  const parts=text.split(/```[a-zA-Z0-9_+-]*\n?([\s\S]*?)```/);
  parts.forEach((p,i)=>{
    if(i%2){const pre=document.createElement("pre"),c=document.createElement("code");c.textContent=p.replace(/\n$/,"");pre.appendChild(c);el.appendChild(pre);return;}
    p.split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/).forEach(t=>{
      if(!t)return;
      if(t.length>2&&t[0]==="`"){const c=document.createElement("code");c.textContent=t.slice(1,-1);el.appendChild(c);}
      else if(t.length>4&&t.startsWith("**")){const b=document.createElement("b");b.textContent=t.slice(2,-2);el.appendChild(b);}
      else el.appendChild(document.createTextNode(t));
    });
  });
}
function add(ev){
  if(ev.kind==="log"){const l=$("log"),near=l.scrollTop+l.clientHeight>=l.scrollHeight-20;l.appendChild(document.createTextNode(ev.text+"\n"));if(near)l.scrollTop=l.scrollHeight;return;}
  const e=$("empty");if(e)e.remove();
  const d=document.createElement("div");d.className="msg "+ev.kind;rich(d,ev.text);
  const c=$("chat"),near=c.scrollTop+c.clientHeight>=c.scrollHeight-80;c.appendChild(d);
  if(near||ev.kind==="user")c.scrollTop=c.scrollHeight;
}
function renderApprovals(list){
  const box=$("approvals"),key=list.map(a=>a.id).join();
  if(box.dataset.key===key)return;box.dataset.key=key;box.textContent="";
  list.forEach(a=>{
    const d=document.createElement("div");d.className="appr";
    const h=document.createElement("b");h.textContent="JARVIS needs your approval";
    const p=document.createElement("pre");p.textContent=a.text;
    const y=document.createElement("button");y.className="primary";y.textContent="Approve";y.onclick=()=>post("/api/approve",{id:a.id,ok:true});
    const n=document.createElement("button");n.className="danger";n.textContent="Deny";n.onclick=()=>post("/api/approve",{id:a.id,ok:false});
    d.append(h,p,y,n);box.appendChild(d);
  });
}
const NAMES={off:"idle",wake:"waiting for “Hey Jarvis”",listening:"listening…",thinking:"thinking…",speaking:"speaking…",mic:"listening (one message)"};
async function poll(){
  try{
    const r=await fetch("/api/state?after="+last);
    if(r.ok){
      const s=await r.json();state=s;
      s.events.forEach(add);last=s.last;
      $("model").textContent=s.model;
      const st=$("status"),label=s.busy&&s.listen!=="speaking"?"working…":(NAMES[s.listen]||s.listen);
      st.textContent=label;st.className="pill"+(s.busy?" busy":(s.listen!=="off"?" on":""));
      $("hf").classList.toggle("on",s.listen!=="off"&&s.listen!=="mic");
      $("hf").textContent=(s.listen!=="off"&&s.listen!=="mic")?"Hands-free: on":"Hands-free";
      $("mic").classList.toggle("on",s.listen==="mic");
      $("voice").classList.toggle("on",s.voice_on);
      $("voice").textContent=s.voice_on?"Voice: on":"Voice: off";
      $("hint").textContent="Voice: "+s.voice+"  |  end a spoken message with “"+s.end_phrase+"”, say “"+s.sleep_phrase+"” to stop hands-free";
      renderApprovals(s.approvals);
    }
  }catch(e){}
  setTimeout(poll,state.busy||(state.listen&&state.listen!=="off")?500:1200);
}
async function send(){
  const t=$("box").value.trim();if(!t)return;
  const r=await post("/api/send",{text:t});
  if(r.status===409){add({kind:"system",text:"JARVIS is still working on the last message."});return;}
  $("box").value="";$("box").style.height="auto";
}
$("send").onclick=send;
$("box").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey&&!e.isComposing){e.preventDefault();send();}});
$("box").addEventListener("input",e=>{e.target.style.height="auto";e.target.style.height=Math.min(e.target.scrollHeight,160)+"px";});
$("hf").onclick=()=>post("/api/listen",{on:!(state.listen&&state.listen!=="off"&&state.listen!=="mic")});
$("mic").onclick=()=>post(state.listen==="mic"?"/api/stopmic":"/api/talk");
$("stopv").onclick=()=>post("/api/stopvoice");
$("voice").onclick=()=>post("/api/send",{text:state.voice_on?"/voice off":"/voice on"});
$("clear").onclick=()=>{post("/api/send",{text:"/clear"});$("chat").textContent="";};
poll();$("box").focus();
</script></body></html>
"""

if __name__ == "__main__":
    main()