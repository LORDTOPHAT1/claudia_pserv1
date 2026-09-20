from flask import Flask, request, jsonify, send_file
from claudia_core import ask_claudia
from importer import process_import
from memory_lib import get_existing_topics
import os
import re
import uuid
import subprocess
import threading
import json
import requests

app = Flask(__name__)
UPLOAD_FOLDER = "/home/linuxuser/uploads"
AUDIO_FOLDER = "/home/linuxuser/audio_cache"
PIPER_BIN = "/home/linuxuser/piper/piper/piper"
PIPER_MODEL = "/home/linuxuser/piper/voices/en_GB-alba-medium.onnx"
WHISPER_SERVER_URL = "http://127.0.0.1:8081/inference"
WHISPER_CLI_BIN = "/home/linuxuser/whisper.cpp/build/bin/whisper-cli"
WHISPER_SMALL_MODEL = "/home/linuxuser/whisper.cpp/models/ggml-small.en.bin"

NON_SPEECH_TAG_RE = re.compile(r"\[[^\[\]]*\]")


def strip_non_speech_tags(text):
    cleaned = NON_SPEECH_TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()


os.makedirs(AUDIO_FOLDER, exist_ok=True)

# --- Logo settings persisted outside the code directory so a redeploy
# (overwriting webapp.py) never clobbers a saved tuning. ---
LOGO_SETTINGS_DIR = "/home/linuxuser/settings"
LOGO_SETTINGS_PATH = os.path.join(LOGO_SETTINGS_DIR, "logo_settings.json")

DEFAULT_LOGO_SETTINGS = {
    "size": 64,
    "minStroke": 1.0,
    "innerPct": 72,
    "angleOffset": 0,
    "baseStroke": 2.0,
    "evenWeights": False,
    "idleSpeed": 8,
    "waitingSpeed": 55,
    "speakingSpeed": 162,
    "speakingSwell": 17,
}

os.makedirs(LOGO_SETTINGS_DIR, exist_ok=True)

FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="-50 -50 100 100" role="img" aria-label="Claudia">
<polygon fill="none" stroke="#a3393d" stroke-width="6.25" stroke-linejoin="round" points="0,-40 34.64,-20 34.64,20 0,40 -34.64,20 -34.64,-20" />
<polygon fill="none" stroke="#a3393d" stroke-width="19.79" stroke-linejoin="round" points="0,-28.8 24.94,-14.4 24.94,14.4 0,28.8 -24.94,14.4 -24.94,-14.4" />
</svg>"""

# --- Piper kept running persistently instead of spawning a fresh process
# per request. --json-input reads one synthesis job per line and keeps
# the model loaded in memory between them.
piper_lock = threading.Lock()


def start_piper_process():
    return subprocess.Popen(
        [PIPER_BIN, "--model", PIPER_MODEL, "--json-input", "-q"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


piper_process = start_piper_process()


def piper_generate(text, filepath):
    global piper_process
    with piper_lock:
        if piper_process.poll() is not None:
            piper_process = start_piper_process()

        piper_process.stdin.write(json.dumps({"text": text, "output_file": filepath}) + "\n")
        piper_process.stdin.flush()
        piper_process.stdout.readline()


PAGE_STYLE = """
:root {
    --bg: #0a0a0b;
    --surface: #131316;
    --surface-2: #1a1a1e;
    --line: #2b2b30;
    --line-bright: #45454c;
    --text: #d6d4cf;
    --text-dim: #7d7d80;
    --text-faint: #4a4a4d;
    --accent: #a3393d;
    --accent-dim: #3a1616;
    --accent-bright: #c25a53;
    --font: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
    --space-1: 0.4em;
    --space-2: 0.8em;
    --space-3: 1.4em;
    --space-4: 2.2em;
    --space-5: 3.4em;
}

* { box-sizing: border-box; }

html, body {
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 15px;
    line-height: 1.55;
}

::selection { background: var(--accent-dim); color: var(--text); }
a { color: var(--accent-bright); }

button, input, select {
    font-family: inherit;
    font-size: inherit;
    color: var(--text);
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: 0;
    padding: 0.5em 0.8em;
}

button {
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    font-size: 0.78em;
    transition: border-color 120ms ease, background-color 120ms ease, color 120ms ease;
}
button:hover:not(:disabled) { border-color: var(--line-bright); }
button:disabled { opacity: 0.45; cursor: default; }

.btn-accent { border-color: var(--accent); color: var(--accent-bright); }
.btn-accent:hover:not(:disabled) { background: var(--accent-dim); }
.btn-small { padding: 0.3em 0.6em; font-size: 0.68em; margin-left: var(--space-2); }

:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.frame {
    display: grid;
    grid-template-columns: 14em 1fr;
    height: 100vh;
}

.sidebar {
    border-right: 1px solid var(--line);
    padding: var(--space-4) var(--space-3);
    background: var(--surface);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.sidebar-head {
    font-size: 0.72em;
    letter-spacing: 0.14em;
    color: var(--text-faint);
    margin-bottom: var(--space-3);
    flex-shrink: 0;
}

.topic-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    overflow-y: auto;
    flex: 1;
    min-height: 0;
}

.topic-item, .topic-empty {
    font-size: 0.82em;
    color: var(--text-dim);
    border-left: 2px solid var(--line);
    padding-left: var(--space-2);
    word-break: break-word;
}
.topic-empty { color: var(--text-faint); }

.sidebar-settings-btn {
    flex-shrink: 0;
    margin-top: var(--space-3);
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid var(--line);
}
.sidebar-settings-btn:hover { border-color: var(--line-bright); color: var(--accent-bright); }

.main {
    width: 100%;
    max-width: 76em;
    margin: 0 auto;
    height: 100vh;
    box-sizing: border-box;
    padding: var(--space-4) var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    overflow: hidden;
}

.header {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-shrink: 0;
    border-bottom: 1px solid var(--line);
    padding-bottom: var(--space-3);
}

.claudia-logo-mount { display: flex; flex-shrink: 0; }
.wordmark { font-size: 1.4em; font-weight: 700; letter-spacing: 0.1em; }

/* --- settings modal --- */

.modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
}
.modal-backdrop[hidden] { display: none; }

.modal {
    background: var(--surface);
    border: 1px solid var(--line-bright);
    padding: var(--space-4);
    width: 100%;
    max-width: 34em;
    max-height: 80vh;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
}

.modal-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid var(--line);
    padding-bottom: var(--space-2);
}
.modal-title { font-size: 0.9em; font-weight: 700; letter-spacing: 0.1em; }
.modal-close {
    background: transparent;
    border: none;
    color: var(--text-dim);
    padding: 0.2em 0.5em;
}
.modal-close:hover { color: var(--accent-bright); }

/* --- chat: one unified frame instead of separately boxed log + input --- */

.chat-frame {
    border: 1px solid var(--line);
    background: var(--surface);
    display: flex;
    flex-direction: column;
    flex: 1;
    min-height: 0;
}

.chat-log {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    border-bottom: 1px solid var(--line);
}

.msg {
    border-left: 2px solid var(--line);
    padding-left: var(--space-2);
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: var(--space-2);
}
.msg-user { border-left-color: var(--line-bright); }
.msg-claudia { border-left-color: var(--accent); }

.msg-role {
    font-size: 0.68em;
    letter-spacing: 0.1em;
    color: var(--text-faint);
    min-width: 5.5em;
}
.msg-claudia .msg-role { color: var(--accent-bright); }

.msg-text { flex: 1 1 20em; white-space: pre-wrap; }

.msg-sources {
    flex-basis: 100%;
    font-size: 0.75em;
    color: var(--text-dim);
    display: flex;
    flex-direction: column;
    gap: 0.2em;
    margin-left: 5.5em;
}

.input-row {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    padding: var(--space-2);
    position: relative;
}
.input-row:focus-within { background: var(--surface-2); }

/* mic icon + device arrow + mode toggle, inline beside the input */

.icon-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.45em;
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-dim);
}
.icon-btn:hover:not(:disabled) { color: var(--text); border-color: var(--line); }
.icon-btn.active { color: var(--accent-bright); }
.icon-btn svg { display: block; }

.mic-control-group { position: relative; display: flex; align-items: center; }

.mic-popover {
    position: absolute;
    bottom: calc(100% + 0.4em);
    right: 0;
    background: var(--surface);
    border: 1px solid var(--line);
    padding: var(--space-2) var(--space-3);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 16em;
    z-index: 10;
}
.mic-popover[hidden] { display: none; }
.field { display: flex; align-items: center; gap: var(--space-2); }
.field-label { font-size: 0.72em; color: var(--text-dim); letter-spacing: 0.05em; }

.mode-toggle { display: flex; border: 1px solid var(--line); flex-shrink: 0; }
.toggle-btn { border: none; background: transparent; color: var(--text-dim); padding: 0.5em 0.7em; }
.toggle-btn + .toggle-btn { border-left: 1px solid var(--line); }
.toggle-btn.active { background: var(--accent-dim); color: var(--accent-bright); }

.input-field-wrap { position: relative; flex: 1; }

.input-field-wrap input[type="text"] {
    width: 100%;
    border: none;
    background: transparent;
    padding: 0.5em 0.6em;
}
.input-field-wrap input[type="text"]:focus { outline: none; }

.input-waveform {
    position: absolute;
    inset: 0;
    pointer-events: none;
    opacity: 0;
    transition: opacity 120ms ease;
}
.input-waveform.active { opacity: 1; }

.input-loader {
    position: absolute;
    left: 0.6em;
    top: 50%;
    transform: translateY(-50%);
    display: none;
}
.input-loader.active { display: inline-block; }

.loader {
    display: inline-block;
    width: 3.2em;
    height: 0.55em;
    background: var(--surface-2);
    border: 1px solid var(--line);
    position: relative;
    overflow: hidden;
    vertical-align: middle;
}
.loader::after {
    content: "";
    position: absolute;
    top: 0; bottom: 0; left: -40%;
    width: 40%;
    background: var(--accent);
    animation: scan 1.1s linear infinite;
}
.loader-inline { width: 2.2em; }

@keyframes scan { 0% { left: -40%; } 100% { left: 100%; } }

@media (prefers-reduced-motion: reduce) {
    .loader::after { animation: none; left: 30%; }
}

section.general-settings, section.import-panel, section.logo-settings {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
}
section.import-panel, section.logo-settings { border-top: 1px solid var(--line); padding-top: var(--space-3); }
.panel-label { font-size: 0.7em; letter-spacing: 0.12em; color: var(--text-faint); }
.status-line { font-size: 0.78em; color: var(--text-dim); margin: 0; }
.upload-row { display: flex; align-items: center; gap: var(--space-2); flex-wrap: wrap; }

/* --- logo settings: sliders on the left, live preview on the right --- */

.logo-settings-body { display: flex; gap: var(--space-4); flex-wrap: wrap; }
.logo-settings-controls { display: flex; flex-direction: column; gap: var(--space-2); flex: 1 1 16em; }
.field-value { font-size: 0.72em; color: var(--text-dim); min-width: 3.4em; text-align: right; flex-shrink: 0; }
.field input[type="range"] { flex: 1; padding: 0; border: none; background: transparent; }

.logo-preview-panel {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-2);
    flex-shrink: 0;
}
.logo-preview-mount {
    width: 140px;
    height: 140px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--line);
    background: var(--surface-2);
}
.logo-preview-buttons { display: flex; gap: var(--space-1); flex-wrap: wrap; justify-content: center; }

@media (max-width: 900px) {
    .frame { grid-template-columns: 1fr; height: auto; }
    .sidebar { border-right: none; border-bottom: 1px solid var(--line); height: auto; max-height: 40vh; }
    .main { padding: var(--space-3); max-width: none; height: auto; min-height: 100vh; overflow: visible; }
    .chat-log { flex: none; height: 60vh; }
}
"""


def render_page(body_html):
    return f"""<!doctype html>
    <html lang="en" data-theme="dark">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Claudia</title>
    <link rel="icon" type="image/svg+xml" href="/favicon.svg">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>{PAGE_STYLE}</style>
    </head>
    <body>
    {body_html}
    </body>
    </html>
    """


@app.route("/")
def home():
    body = """
        <div class="frame">
            <aside class="sidebar">
                <div class="sidebar-head">TOPICS</div>
                <ul id="topic-list" class="topic-list"><li class="topic-empty">loading&hellip;</li></ul>
                <button class="sidebar-settings-btn" onclick="openSettingsModal()" type="button">SETTINGS</button>
            </aside>

            <main class="main">
                <header class="header">
                    <div id="claudia-logo-mount" class="claudia-logo-mount"></div>
                    <div class="wordmark">CLAUDIA</div>
                </header>

                <div class="chat-frame">
                    <div id="chat-log" class="chat-log"></div>

                    <div class="input-row">
                        <div class="input-field-wrap">
                            <input type="text" id="message-input" placeholder="say something" autocomplete="off">
                            <canvas id="input-waveform" class="input-waveform"></canvas>
                            <span id="input-loader" class="loader input-loader"></span>
                        </div>

                        <div class="mic-control-group">
                            <button class="icon-btn" id="mic-icon-btn" onclick="handleMicClick()" aria-label="toggle microphone" type="button">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                                    <line x1="12" y1="19" x2="12" y2="23"></line>
                                    <line x1="8" y1="23" x2="16" y2="23"></line>
                                </svg>
                            </button>

                            <button class="icon-btn" id="mic-arrow-btn" onclick="toggleMicPopover(event)" aria-label="microphone options" type="button">
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                    <polyline points="6 9 12 15 18 9"></polyline>
                                </svg>
                            </button>

                            <div class="mic-popover" id="mic-popover" hidden>
                                <label class="field">
                                    <span class="field-label">device</span>
                                    <select id="mic-device-select"></select>
                                </label>
                            </div>
                        </div>

                        <div class="mode-toggle" role="radiogroup" aria-label="listening mode">
                            <button type="button" id="mode-live-btn" class="toggle-btn active" onclick="setSttMode('live')">LIVE</button>
                            <button type="button" id="mode-accurate-btn" class="toggle-btn" onclick="setSttMode('accurate')">ACC</button>
                        </div>

                        <button class="btn-accent" onclick="sendMessage()" type="button">SEND</button>
                    </div>
                </div>
            </main>
        </div>

        <div class="modal-backdrop" id="settings-modal-backdrop" hidden onclick="if (event.target === this) closeSettingsModal()">
            <div class="modal">
                <div class="modal-head">
                    <div class="modal-title">SETTINGS</div>
                    <button class="modal-close" onclick="closeSettingsModal()" type="button" aria-label="close settings">&times;</button>
                </div>

                <section class="general-settings">
                    <div class="panel-label">SYSTEM</div>
                    <p class="status-line">reserved &mdash; general settings not yet wired in</p>
                </section>

                <section class="logo-settings">
                    <div class="panel-label">LOGO</div>

                    <div class="logo-settings-body">
                        <div class="logo-settings-controls">
                            <label class="field">
                                <span class="field-label">size</span>
                                <input type="range" id="logo-size" min="24" max="128" step="1">
                                <span class="field-value" id="logo-size-val"></span>
                            </label>
                            <label class="field">
                                <span class="field-label">min stroke</span>
                                <input type="range" id="logo-minStroke" min="0.5" max="4" step="0.1">
                                <span class="field-value" id="logo-minStroke-val"></span>
                            </label>
                            <label class="field">
                                <span class="field-label">inner size</span>
                                <input type="range" id="logo-innerPct" min="40" max="100" step="0.1">
                                <span class="field-value" id="logo-innerPct-val"></span>
                            </label>
                            <label class="field">
                                <span class="field-label">angle offset</span>
                                <input type="range" id="logo-angleOffset" min="0" max="30" step="1">
                                <span class="field-value" id="logo-angleOffset-val"></span>
                            </label>
                            <label class="field">
                                <span class="field-label">stroke</span>
                                <input type="range" id="logo-baseStroke" min="0.6" max="4" step="0.1">
                                <span class="field-value" id="logo-baseStroke-val"></span>
                            </label>
                            <label class="field">
                                <span class="field-label">idle speed</span>
                                <input type="range" id="logo-idleSpeed" min="0" max="60" step="1">
                                <span class="field-value" id="logo-idleSpeed-val"></span>
                            </label>
                            <label class="field">
                                <span class="field-label">waiting speed</span>
                                <input type="range" id="logo-waitingSpeed" min="10" max="200" step="1">
                                <span class="field-value" id="logo-waitingSpeed-val"></span>
                            </label>
                            <label class="field">
                                <span class="field-label">speaking speed</span>
                                <input type="range" id="logo-speakingSpeed" min="40" max="360" step="1">
                                <span class="field-value" id="logo-speakingSpeed-val"></span>
                            </label>
                            <label class="field">
                                <span class="field-label">speaking swell</span>
                                <input type="range" id="logo-speakingSwell" min="0" max="40" step="1">
                                <span class="field-value" id="logo-speakingSwell-val"></span>
                            </label>

                            <div class="mode-toggle" role="radiogroup" aria-label="stroke weights">
                                <button type="button" id="logo-weight-tapered-btn" class="toggle-btn active" onclick="setLogoWeightMode(false)">TAPERED</button>
                                <button type="button" id="logo-weight-even-btn" class="toggle-btn" onclick="setLogoWeightMode(true)">EVEN</button>
                            </div>
                        </div>

                        <div class="logo-preview-panel">
                            <div id="logo-preview-mount" class="logo-preview-mount"></div>
                            <div class="logo-preview-buttons">
                                <button type="button" class="btn-small" onclick="previewLogoState('idle')">IDLE</button>
                                <button type="button" class="btn-small" onclick="previewLogoState('waiting')">WAITING</button>
                                <button type="button" class="btn-small" onclick="previewLogoState('speaking')">SPEAKING</button>
                                <button type="button" class="btn-small" onclick="previewLogoState('static')">STATIC</button>
                            </div>
                        </div>
                    </div>

                    <div class="upload-row">
                        <button type="button" class="btn-accent" onclick="saveLogoSettings()">SAVE</button>
                        <button type="button" onclick="resetLogoSettings()">RESET TO DEFAULTS</button>
                        <span class="status-line" id="logo-settings-status"></span>
                    </div>
                </section>

                <section class="import-panel">
                    <div class="panel-label">IMPORT CONVERSATION</div>
                    <form action="/upload" method="post" enctype="multipart/form-data" class="upload-row">
                        <input type="file" name="file" accept=".json" multiple>
                        <button type="submit">UPLOAD</button>
                    </form>
                </section>
            </div>
        </div>

        <script>
        function openSettingsModal() {
            document.getElementById("settings-modal-backdrop").hidden = false;
        }
        function closeSettingsModal() {
            document.getElementById("settings-modal-backdrop").hidden = true;
        }
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape") closeSettingsModal();
        });

        // --- Claudia logo: two hexagons turning opposite ways, plain SVG,
        // no deps. One shared rAF loop drives every instance on the page. ---

        const CLAUDIA_LOGOS = [];
        let claudiaLoopStarted = false;
        let claudiaReduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        window.matchMedia("(prefers-reduced-motion: reduce)").addEventListener("change", (e) => {
            claudiaReduceMotion = e.matches;
        });

        function claudiaHexPoints(radius, rotationDeg) {
            const pts = [];
            for (let i = 0; i < 6; i++) {
                const a = (rotationDeg + 60 * i) * Math.PI / 180;
                pts.push((radius * Math.cos(a)) + "," + (radius * Math.sin(a)));
            }
            return pts.join(" ");
        }

        function claudiaLoop(now) {
            requestAnimationFrame(claudiaLoop);
            if (!claudiaLoop.last) claudiaLoop.last = now;
            const dt = Math.min(0.05, (now - claudiaLoop.last) / 1000);
            claudiaLoop.last = now;
            CLAUDIA_LOGOS.forEach(logo => logo._tick(dt));
        }

        let claudiaGradSeq = 0;

        function createClaudiaLogo(container, opts) {
            const cfg = Object.assign({
                size: 64, minStroke: 1.0, innerPct: 72, angleOffset: 0, baseStroke: 2.0,
                evenWeights: false, idleSpeed: 8, waitingSpeed: 55, speakingSpeed: 162,
                speakingSwell: 17, color: "gradient",
            }, opts || {});

            const gradId = "claudia-grad-" + (claudiaGradSeq++);
            const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            svg.setAttribute("viewBox", "-50 -50 100 100");
            svg.setAttribute("role", "img");
            svg.setAttribute("aria-label", "Claudia");
            svg.innerHTML = `<defs><linearGradient id="${gradId}" gradientUnits="userSpaceOnUse" x1="-40" y1="-40" x2="40" y2="40">
                <stop offset="0%" stop-color="#8a2c33" /><stop offset="100%" stop-color="#e5563c" /></linearGradient></defs>
                <polygon class="claudia-outer" fill="none" stroke-linejoin="round" />
                <polygon class="claudia-inner" fill="none" stroke-linejoin="round" />`;
            const outerEl = svg.querySelector(".claudia-outer");
            const innerEl = svg.querySelector(".claudia-inner");
            const stroke = cfg.color === "gradient" ? `url(#${gradId})` : cfg.color;
            outerEl.setAttribute("stroke", stroke);
            innerEl.setAttribute("stroke", stroke);
            container.innerHTML = "";
            container.appendChild(svg);

            const st = { mode: "idle", t: 0, angOuter: -90, angInner: -90 + cfg.angleOffset,
                speed: cfg.idleSpeed, scaleOuter: 1, scaleInner: 1, amp: 0, targetAmp: 0 };

            function render() {
                svg.setAttribute("width", cfg.size);
                svg.setAttribute("height", cfg.size);
                const outerUnits = cfg.baseStroke * (cfg.evenWeights ? 1 : 0.6);
                const innerUnits = cfg.baseStroke * (cfg.evenWeights ? 1 : 1.9);
                const k = Math.max(1, cfg.minStroke / (outerUnits * cfg.size / 100));
                outerEl.setAttribute("points", claudiaHexPoints(40, st.angOuter));
                innerEl.setAttribute("points", claudiaHexPoints(40 * cfg.innerPct / 100, st.angInner));
                outerEl.setAttribute("transform", `scale(${st.scaleOuter})`);
                innerEl.setAttribute("transform", `scale(${st.scaleInner})`);
                outerEl.setAttribute("stroke-width", outerUnits * k);
                innerEl.setAttribute("stroke-width", innerUnits * k);
            }

            function snapToStatic() {
                st.angOuter = -90;
                st.angInner = -90 + cfg.angleOffset;
                st.speed = 0;
                st.scaleOuter = 1;
                st.scaleInner = 1;
                render();
            }

            const logo = {
                setState(next) {
                    st.mode = next;
                    if (next === "static") snapToStatic();
                },
                setAmplitude(v) { st.targetAmp = Math.max(0, Math.min(1, v)); },
                updateConfig(next) {
                    Object.assign(cfg, next);
                    if (st.mode === "static") snapToStatic(); else render();
                },
                _tick(dt) {
                    if (claudiaReduceMotion || st.mode === "static") { snapToStatic(); return; }
                    st.t += dt;
                    st.amp += (st.targetAmp - st.amp) * Math.min(1, dt * 12);

                    let targetSpeed, targetOuter, targetInner;
                    if (st.mode === "waiting") {
                        targetSpeed = cfg.waitingSpeed;
                        targetOuter = 1 + 0.06 * Math.sin(5 * st.t);
                        targetInner = 1 + 0.06 * Math.sin(5 * st.t + Math.PI);
                    } else if (st.mode === "speaking") {
                        targetSpeed = 14 + st.amp * (cfg.speakingSpeed - 14);
                        targetOuter = targetInner = 1 + st.amp * (cfg.speakingSwell / 100);
                    } else {
                        targetSpeed = cfg.idleSpeed;
                        targetOuter = 1 + 0.03 * Math.sin(1.6 * st.t);
                        targetInner = 1 + 0.03 * Math.sin(1.6 * st.t + 1);
                    }

                    st.speed += (targetSpeed - st.speed) * Math.min(1, dt * 5);
                    st.scaleOuter += (targetOuter - st.scaleOuter) * Math.min(1, dt * 8);
                    st.scaleInner += (targetInner - st.scaleInner) * Math.min(1, dt * 8);
                    st.angOuter += st.speed * dt;
                    st.angInner -= st.speed * dt;
                    render();
                },
            };

            render();
            CLAUDIA_LOGOS.push(logo);
            if (!claudiaLoopStarted) { claudiaLoopStarted = true; requestAnimationFrame(claudiaLoop); }
            return logo;
        }

        // --- Waveform visualizer: a single moving line, no grid/background,
        // shared between live and accurate mode since only one captures at a time ---

        function createWaveformVisualizer(canvasId) {
            let audioContext, analyser, animationId;
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext("2d");

            function sizeToParent() {
                const rect = canvas.parentElement.getBoundingClientRect();
                canvas.width = rect.width;
                canvas.height = rect.height;
            }

            return {
                start(stream) {
                    sizeToParent();
                    canvas.classList.add("active");

                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const source = audioContext.createMediaStreamSource(stream);
                    analyser = audioContext.createAnalyser();
                    analyser.fftSize = 256;
                    source.connect(analyser);

                    const bufferLength = analyser.frequencyBinCount;
                    const dataArray = new Uint8Array(bufferLength);

                    function draw() {
                        animationId = requestAnimationFrame(draw);
                        analyser.getByteTimeDomainData(dataArray);

                        ctx.fillStyle = "#1a1a1e";
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        ctx.lineWidth = 1.5;
                        ctx.strokeStyle = "#a3393d";
                        ctx.beginPath();

                        const sliceWidth = canvas.width / bufferLength;
                        let x = 0;
                        for (let i = 0; i < bufferLength; i++) {
                            const v = dataArray[i] / 128.0;
                            const y = (v * canvas.height) / 2;
                            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                            x += sliceWidth;
                        }
                        ctx.lineTo(canvas.width, canvas.height / 2);
                        ctx.stroke();
                    }
                    draw();
                },
                stop() {
                    if (animationId) cancelAnimationFrame(animationId);
                    if (audioContext) audioContext.close();
                    canvas.classList.remove("active");
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
            };
        }

        const inputWaveform = createWaveformVisualizer("input-waveform");

        // --- mode + mic control dispatch ---

        let currentMode = "live";

        function setSttMode(mode) {
            currentMode = mode;
            document.getElementById("mode-live-btn").classList.toggle("active", mode === "live");
            document.getElementById("mode-accurate-btn").classList.toggle("active", mode === "accurate");
        }

        function toggleMicPopover(e) {
            e.stopPropagation();
            const pop = document.getElementById("mic-popover");
            pop.hidden = !pop.hidden;
        }
        document.addEventListener("click", (e) => {
            const pop = document.getElementById("mic-popover");
            if (!pop.hidden && !pop.contains(e.target)) pop.hidden = true;
        });

        function setMicChromeBusy(busy) {
            document.getElementById("mic-arrow-btn").disabled = busy;
            document.getElementById("mode-live-btn").disabled = busy;
            document.getElementById("mode-accurate-btn").disabled = busy;
        }

        function handleMicClick() {
            if (currentMode === "live") {
                toggleListening();
            } else {
                toggleAccurateRecording();
            }
        }

        // --- Live mode: rolling 3s chunks against /listen (base.en, always-on server) ---

        let listening = false;
        let currentStream;
        let activeRecorder = null;
        let cycleTimer = null;

        async function toggleListening() {
            const icon = document.getElementById("mic-icon-btn");

            if (!listening) {
                listening = true;
                icon.classList.add("active");
                setMicChromeBusy(true);

                currentStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                inputWaveform.start(currentStream);
                startNewRecorder();
                runCycle();
            } else {
                listening = false;
                icon.classList.remove("active");
                setMicChromeBusy(false);

                if (cycleTimer) clearTimeout(cycleTimer);
                if (activeRecorder && activeRecorder.state === "recording") {
                    activeRecorder.stop();
                }
                if (currentStream) {
                    currentStream.getTracks().forEach(track => track.stop());
                }
                inputWaveform.stop();
            }
        }

        function startNewRecorder() {
            const recorder = new MediaRecorder(currentStream);
            const chunks = [];

            recorder.ondataavailable = (event) => {
                if (event.data.size > 0) chunks.push(event.data);
            };

            recorder.onstop = async () => {
                const blob = new Blob(chunks, { type: "audio/webm" });
                sendChunk(blob);
            };

            recorder.start();
            activeRecorder = recorder;
        }

        function runCycle() {
            if (!listening) return;

            cycleTimer = setTimeout(() => {
                const finishingRecorder = activeRecorder;
                startNewRecorder();
                if (finishingRecorder && finishingRecorder.state === "recording") {
                    finishingRecorder.stop();
                }
                runCycle();
            }, 3000);
        }

        async function sendChunk(blob) {
            const formData = new FormData();
            formData.append("audio", blob, "chunk.webm");

            const response = await fetch("/listen", { method: "POST", body: formData });
            const data = await response.json();

            if (data.text && data.text.trim()) {
                const input = document.getElementById("message-input");
                input.value = (input.value + " " + data.text.trim()).trim();
            }
        }

        // --- Settings: mic device list, topics sidebar ---

        async function populateMicDevices() {
            try {
                const tempStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                tempStream.getTracks().forEach(t => t.stop());
            } catch (e) {
                console.error("Mic permission denied", e);
            }

            const devices = await navigator.mediaDevices.enumerateDevices();
            const select = document.getElementById("mic-device-select");
            select.innerHTML = "";
            devices.filter(d => d.kind === "audioinput").forEach(d => {
                const opt = document.createElement("option");
                opt.value = d.deviceId;
                opt.innerText = d.label || ("microphone " + (select.length + 1));
                select.appendChild(opt);
            });
        }

        async function loadTopics() {
            const list = document.getElementById("topic-list");
            try {
                const response = await fetch("/topics");
                const data = await response.json();
                list.innerHTML = "";

                if (!data.topics || data.topics.length === 0) {
                    const li = document.createElement("li");
                    li.className = "topic-empty";
                    li.innerText = "no topics yet";
                    list.appendChild(li);
                    return;
                }

                data.topics.forEach(t => {
                    const li = document.createElement("li");
                    li.className = "topic-item";
                    li.innerText = t;
                    list.appendChild(li);
                });
            } catch (e) {
                console.error("Failed to load topics", e);
                list.innerHTML = "<li class='topic-empty'>unavailable</li>";
            }
        }

        // --- Logo settings panel: sliders drive a live preview instance and
        // the header instance identically, then persist to the server ---

        const LOGO_DEFAULTS = {
            size: 64, minStroke: 1.0, innerPct: 72, angleOffset: 0, baseStroke: 2.0,
            evenWeights: false, idleSpeed: 8, waitingSpeed: 55, speakingSpeed: 162, speakingSwell: 17,
        };
        const LOGO_FIELDS = ["size", "minStroke", "innerPct", "angleOffset", "baseStroke",
            "idleSpeed", "waitingSpeed", "speakingSpeed", "speakingSwell"];
        const LOGO_FIELD_UNITS = {
            size: "px", minStroke: "px", innerPct: "%", angleOffset: "°", baseStroke: "u",
            idleSpeed: "°/s", waitingSpeed: "°/s", speakingSpeed: "°/s", speakingSwell: "%",
        };

        let headerLogo, previewLogo, previewAmpTimer;
        let workingLogoSettings = Object.assign({}, LOGO_DEFAULTS);

        function applyLogoSettingsToForm(settings) {
            LOGO_FIELDS.forEach(key => {
                document.getElementById("logo-" + key).value = settings[key];
                document.getElementById("logo-" + key + "-val").innerText = settings[key] + LOGO_FIELD_UNITS[key];
            });
            document.getElementById("logo-weight-tapered-btn").classList.toggle("active", !settings.evenWeights);
            document.getElementById("logo-weight-even-btn").classList.toggle("active", !!settings.evenWeights);
        }

        function onLogoSettingsInput() {
            LOGO_FIELDS.forEach(key => {
                workingLogoSettings[key] = parseFloat(document.getElementById("logo-" + key).value);
                document.getElementById("logo-" + key + "-val").innerText = workingLogoSettings[key] + LOGO_FIELD_UNITS[key];
            });
            if (previewLogo) previewLogo.updateConfig(workingLogoSettings);
            if (headerLogo) headerLogo.updateConfig(workingLogoSettings);
        }

        function setLogoWeightMode(even) {
            workingLogoSettings.evenWeights = even;
            document.getElementById("logo-weight-tapered-btn").classList.toggle("active", !even);
            document.getElementById("logo-weight-even-btn").classList.toggle("active", even);
            if (previewLogo) previewLogo.updateConfig(workingLogoSettings);
            if (headerLogo) headerLogo.updateConfig(workingLogoSettings);
        }

        function previewLogoState(state) {
            if (previewAmpTimer) { clearInterval(previewAmpTimer); previewAmpTimer = null; }
            previewLogo.setState(state);
            if (state === "speaking") {
                previewAmpTimer = setInterval(() => previewLogo.setAmplitude(0.25 + Math.random() * 0.75), 90);
            } else {
                previewLogo.setAmplitude(0);
            }
        }

        async function saveLogoSettings() {
            const status = document.getElementById("logo-settings-status");
            status.innerText = "saving…";
            try {
                await fetch("/logo-settings", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify(workingLogoSettings),
                });
                status.innerText = "saved";
            } catch (e) {
                status.innerText = "save failed";
            }
            setTimeout(() => { status.innerText = ""; }, 2000);
        }

        function resetLogoSettings() {
            workingLogoSettings = Object.assign({}, LOGO_DEFAULTS);
            applyLogoSettingsToForm(workingLogoSettings);
            if (previewLogo) previewLogo.updateConfig(workingLogoSettings);
            if (headerLogo) headerLogo.updateConfig(workingLogoSettings);
        }

        async function initLogo() {
            try {
                const response = await fetch("/logo-settings");
                workingLogoSettings = Object.assign({}, LOGO_DEFAULTS, await response.json());
            } catch (e) {
                workingLogoSettings = Object.assign({}, LOGO_DEFAULTS);
            }

            applyLogoSettingsToForm(workingLogoSettings);
            LOGO_FIELDS.forEach(key => {
                document.getElementById("logo-" + key).addEventListener("input", onLogoSettingsInput);
            });

            headerLogo = createClaudiaLogo(document.getElementById("claudia-logo-mount"),
                Object.assign({}, workingLogoSettings, { color: "gradient" }));
            headerLogo.setState("idle");

            previewLogo = createClaudiaLogo(document.getElementById("logo-preview-mount"),
                Object.assign({}, workingLogoSettings, { color: "gradient" }));
            previewLogo.setState("idle");
        }

        window.addEventListener("load", () => {
            populateMicDevices();
            loadTopics();
            initLogo();
        });

        // --- Accurate mode: full-length recording / upload, one-shot transcription ---

        let accurateRecording = false;
        let accurateStream;
        let accurateRecorder;
        let accurateChunks = [];

        async function toggleAccurateRecording() {
            const icon = document.getElementById("mic-icon-btn");

            if (!accurateRecording) {
                accurateRecording = true;
                icon.classList.add("active");
                setMicChromeBusy(true);

                const deviceId = document.getElementById("mic-device-select").value;
                const constraints = deviceId ? { audio: { deviceId: { exact: deviceId } } } : { audio: true };
                accurateStream = await navigator.mediaDevices.getUserMedia(constraints);
                inputWaveform.start(accurateStream);

                accurateChunks = [];
                accurateRecorder = new MediaRecorder(accurateStream);
                accurateRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) accurateChunks.push(e.data);
                };
                accurateRecorder.onstop = async () => {
                    const blob = new Blob(accurateChunks, { type: "audio/webm" });
                    await submitAccurateAudio(blob, "recording.webm");
                };
                accurateRecorder.start();
            } else {
                accurateRecording = false;
                icon.classList.remove("active");
                icon.disabled = true;

                if (accurateRecorder && accurateRecorder.state === "recording") {
                    accurateRecorder.stop();
                }
                if (accurateStream) {
                    accurateStream.getTracks().forEach(t => t.stop());
                }
                inputWaveform.stop();
            }
        }

        async function submitAccurateAudio(fileOrBlob, filename) {
            const formData = new FormData();
            formData.append("audio", fileOrBlob, filename);
            const icon = document.getElementById("mic-icon-btn");
            const input = document.getElementById("message-input");
            const loader = document.getElementById("input-loader");
            loader.classList.add("active");

            try {
                const response = await fetch("/listen_accurate", { method: "POST", body: formData });
                const data = await response.json();
                if (data.text && data.text.trim()) {
                    input.value = (input.value + " " + data.text.trim()).trim();
                } else {
                    const original = input.placeholder;
                    input.placeholder = "no speech detected";
                    setTimeout(() => { input.placeholder = original; }, 3000);
                }
            } catch (e) {
                console.error(e);
                const original = input.placeholder;
                input.placeholder = "error transcribing";
                setTimeout(() => { input.placeholder = original; }, 3000);
            } finally {
                loader.classList.remove("active");
                icon.disabled = false;
                setMicChromeBusy(false);
            }
        }

        // --- Chat: message rendering, thinking indicator, read-aloud with a loading guard ---

        function appendUserMessage(text) {
            const log = document.getElementById("chat-log");
            const row = document.createElement("div");
            row.className = "msg msg-user";

            const role = document.createElement("span");
            role.className = "msg-role";
            role.innerText = "YOU";

            const body = document.createElement("span");
            body.className = "msg-text";
            body.innerText = text;

            row.appendChild(role);
            row.appendChild(body);
            log.appendChild(row);
            log.scrollTop = log.scrollHeight;
        }

        function appendThinkingRow() {
            const log = document.getElementById("chat-log");
            const row = document.createElement("div");
            row.className = "msg msg-claudia";

            const role = document.createElement("span");
            role.className = "msg-role";
            role.innerText = "CLAUDIA";

            const loader = document.createElement("span");
            loader.className = "loader";

            row.appendChild(role);
            row.appendChild(loader);
            log.appendChild(row);
            log.scrollTop = log.scrollHeight;
            return row;
        }

        function appendClaudiaMessage(replyText, sources) {
            const log = document.getElementById("chat-log");
            const replyId = "reply-" + Date.now();

            const row = document.createElement("div");
            row.className = "msg msg-claudia";

            const role = document.createElement("span");
            role.className = "msg-role";
            role.innerText = "CLAUDIA";

            const body = document.createElement("span");
            body.className = "msg-text";
            body.id = replyId;
            body.innerText = replyText;

            const speakBtn = document.createElement("button");
            speakBtn.className = "btn-small";
            speakBtn.innerText = "READ ALOUD";
            speakBtn.onclick = () => speak(replyId, speakBtn);

            row.appendChild(role);
            row.appendChild(body);
            row.appendChild(speakBtn);

            if (sources && sources.length > 0) {
                const srcDiv = document.createElement("div");
                srcDiv.className = "msg-sources";
                sources.forEach(src => {
                    const a = document.createElement("a");
                    a.href = src;
                    a.target = "_blank";
                    a.rel = "noopener";
                    a.innerText = src;
                    srcDiv.appendChild(a);
                });
                row.appendChild(srcDiv);
            }

            log.appendChild(row);
            log.scrollTop = log.scrollHeight;
            window["text-" + replyId] = replyText;
        }

        async function sendMessage() {
            const input = document.getElementById("message-input");
            const message = input.value.trim();
            if (!message) return;

            appendUserMessage(message);
            input.value = "";

            const thinkingRow = appendThinkingRow();
            if (headerLogo) headerLogo.setState("waiting");

            let data;
            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({message: message})
                });
                data = await response.json();
            } catch (e) {
                thinkingRow.remove();
                if (headerLogo) headerLogo.setState("idle");
                appendClaudiaMessage("(error reaching claudia)", []);
                return;
            }

            thinkingRow.remove();
            if (headerLogo) headerLogo.setState("idle");
            appendClaudiaMessage(data.reply, data.sources || []);
        }

        // --- Speaking amplitude: route Read aloud playback through the Web
        // Audio API so the logo's speaking state follows the real voice
        // instead of a fake animation. ---

        let claudiaAudioCtx;

        function attachSpeakingAnalyser(audio) {
            if (!claudiaAudioCtx) claudiaAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const source = claudiaAudioCtx.createMediaElementSource(audio);
            const analyser = claudiaAudioCtx.createAnalyser();
            analyser.fftSize = 512;
            source.connect(analyser);
            analyser.connect(claudiaAudioCtx.destination);

            const data = new Uint8Array(analyser.fftSize);
            let rafId, stopped = false;

            function pump() {
                analyser.getByteTimeDomainData(data);
                let sumSquares = 0;
                for (let i = 0; i < data.length; i++) {
                    const v = (data[i] - 128) / 128;
                    sumSquares += v * v;
                }
                const rms = Math.sqrt(sumSquares / data.length);
                if (headerLogo) headerLogo.setAmplitude(Math.min(1, rms * 4));
                rafId = requestAnimationFrame(pump);
            }

            function stopTracking() {
                if (stopped) return;
                stopped = true;
                cancelAnimationFrame(rafId);
                if (headerLogo) { headerLogo.setAmplitude(0); headerLogo.setState("idle"); }
            }

            audio.addEventListener("ended", stopTracking);
            audio.addEventListener("pause", stopTracking);
            pump();
        }

        // speakButton is disabled and shown as a loader for the duration of the
        // request so a second click during generation can't fire a second,
        // overlapping playback.
        async function speak(replyId, speakButton) {
            const text = window["text-" + replyId];
            if (!text) return;

            speakButton.disabled = true;
            const originalLabel = speakButton.innerText;
            speakButton.innerHTML = "<span class='loader loader-inline'></span>";

            try {
                const response = await fetch("/speak", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({text: text})
                });
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const audio = new Audio(url);
                attachSpeakingAnalyser(audio);
                if (headerLogo) headerLogo.setState("speaking");
                audio.play();
            } catch (e) {
                console.error("Speak error", e);
                if (headerLogo) headerLogo.setState("idle");
            } finally {
                speakButton.disabled = false;
                speakButton.innerText = originalLabel;
            }
        }

        document.getElementById("message-input").addEventListener("keypress", function(e) {
            if (e.key === "Enter") sendMessage();
        });
        </script>
    """
    return render_page(body)


@app.route("/topics")
def topics():
    return jsonify({"topics": get_existing_topics()})

@app.route("/favicon.svg")
def favicon():
    return app.response_class(FAVICON_SVG, mimetype="image/svg+xml")

@app.route("/logo-settings", methods=["GET"])
def get_logo_settings():
    if os.path.exists(LOGO_SETTINGS_PATH):
        with open(LOGO_SETTINGS_PATH) as f:
            saved = json.load(f)
        merged = dict(DEFAULT_LOGO_SETTINGS)
        merged.update(saved)
        return jsonify(merged)
    return jsonify(DEFAULT_LOGO_SETTINGS)

@app.route("/logo-settings", methods=["POST"])
def save_logo_settings():
    data = request.get_json() or {}
    merged = dict(DEFAULT_LOGO_SETTINGS)
    merged.update({k: v for k, v in data.items() if k in DEFAULT_LOGO_SETTINGS})
    with open(LOGO_SETTINGS_PATH, "w") as f:
        json.dump(merged, f)
    return jsonify(merged)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"reply": "Say something first.", "sources": []})

    result = ask_claudia(user_message)
    return jsonify(result)

@app.route("/listen", methods=["POST"])
def listen():
    audio_file = request.files.get("audio")

    if not audio_file:
        return jsonify({"text": ""})

    temp_path = os.path.join(AUDIO_FOLDER, f"chunk_{uuid.uuid4().hex}.webm")
    audio_file.save(temp_path)

    try:
        with open(temp_path, "rb") as f:
            response = requests.post(
                WHISPER_SERVER_URL,
                files={"file": f},
                data={"response_format": "json"}
            )
        result = response.json()
        text = strip_non_speech_tags(result.get("text", ""))
    except Exception as e:
        print(f"Whisper server error: {e}")
        text = ""
    finally:
        os.remove(temp_path)

    return jsonify({"text": text})

@app.route("/listen_accurate", methods=["POST"])
def listen_accurate():
    audio_file = request.files.get("audio")

    if not audio_file:
        return jsonify({"text": ""})

    raw_path = os.path.join(AUDIO_FOLDER, f"accurate_{uuid.uuid4().hex}_raw")
    wav_path = raw_path + ".wav"
    audio_file.save(raw_path)

    text = ""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        result = subprocess.run(
            [WHISPER_CLI_BIN, "-m", WHISPER_SMALL_MODEL, "-f", wav_path, "-nt", "-np"],
            capture_output=True, text=True, check=True
        )
        text = strip_non_speech_tags(result.stdout)
    except Exception as e:
        print(f"Accurate STT error: {e}")
        text = ""
    finally:
        for p in (raw_path, wav_path):
            if os.path.exists(p):
                os.remove(p)

    return jsonify({"text": text})

@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return "No text provided", 400

    filename = f"speech_{uuid.uuid4().hex}.wav"
    filepath = os.path.join(AUDIO_FOLDER, filename)

    piper_generate(text, filepath)

    return send_file(filepath, mimetype="audio/wav")

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("file")

    if not files:
        return "No files were uploaded."

    results = []
    for f in files:
        if f.filename:
            filepath = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(filepath)
            result = process_import(filepath)
            results.append(result)

    return "<br>".join(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
