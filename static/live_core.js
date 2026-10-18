"use strict";
(function () {
/* 7 Live — pure UI logic (no DOM access at load time).
 *
 * Shared by the browser page (static/live.html, classic <script>) and by the
 * M3 exam under node:
 *
 *   const core = require("./live_core.js");
 *   core.mergeCaption([], "seven", "hello");
 *
 * Exports
 *   STATES                     the 8 orb/conversation states, in order
 *   STATE_STYLE[state]         {color, speed, pulse} — 8 distinct colours
 *   STATE_LABEL[state]         UI label for the page (labelFor maps any internal state)
 *   mergeCaption(lines, who, text)  -> new lines array (pure; never mutates)
 *   endTurn(lines)                  -> new array with every line closed
 *   orbParams(state, level, t) -> {color, radius, spin, particles}
 *   duckVolume(state, base, userSpeaking) -> effective music volume 0..1
 */

/* ---- states ------------------------------------------------------------- */

const STATES = [
  "idle",
  "listening",
  "thinking",
  "executing",
  "speaking",
  "muted",
  "error",
  "reconnecting",
];

/* One distinct hue per state. speed = orbit speed multiplier (>0),
 * pulse = breathing strength of the core glow (>=0). */
const STATE_STYLE = {
  idle:         { color: "#4f6b8a", speed: 0.35, pulse: 0.10 }, // slate blue
  listening:    { color: "#38bdf8", speed: 0.80, pulse: 0.30 }, // sky
  thinking:     { color: "#a78bfa", speed: 1.60, pulse: 0.55 }, // violet
  executing:    { color: "#fbbf24", speed: 1.30, pulse: 0.40 }, // amber
  speaking:     { color: "#34d399", speed: 1.00, pulse: 0.65 }, // emerald
  muted:        { color: "#8d99ae", speed: 0.18, pulse: 0.05 }, // grey
  error:        { color: "#f87171", speed: 0.50, pulse: 0.80 }, // red
  reconnecting: { color: "#fb923c", speed: 1.90, pulse: 0.45 }, // orange
};

/* ---- UI labels (M15a): truthful, human words for what 7 is doing ----------
 * The page shows these in #state and mirrors them to body[data-state], which
 * the CSS uses to colour the orb, the state dot and the word. The internal
 * websocket states stay the same; labelFor() maps them:
 *   idle (before Start)  -> ready       "Ready — tap Start to talk"
 *   listening            -> listening   "Listening"
 *   thinking / executing -> thinking    "Thinking"
 *   speaking             -> speaking    "Speaking"
 *   muted                -> muted       "Muted"
 *   paused (M13)         -> paused      "Paused"
 *   reconnecting         -> reconnecting "Reconnecting"
 *   error                -> error       "Something went wrong"
 */

const STATE_LABEL = {
  ready: "ready",
  listening: "listening",
  thinking: "thinking",
  speaking: "speaking",
  muted: "muted",
  paused: "paused",
  reconnecting: "reconnecting",
  error: "error",
};

const STATE_WORD = {
  ready: "Ready — tap Start to talk",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
  muted: "Muted",
  paused: "Paused",
  reconnecting: "Reconnecting",
  error: "Something went wrong",
};

function labelFor(state) {
  if (state === "idle") return "ready";
  if (state === "executing") return "thinking";
  return STATE_LABEL[state] || "ready";
}

/* ---- captions ------------------------------------------------------------
 *
 * lines is an array of {who, text, open}. Fragments from the live API are
 * appended to the last line while it is open and from the same speaker;
 * otherwise a new line is pushed. Fragments are trimmed and joined with ONE
 * space, except: no space before , . ! ? ; : ' ) and no space after (.
 * Empty / whitespace-only text is ignored. Never mutates the input.
 */

const NO_SPACE_BEFORE = /^[,\.\!\?;:'\)]/;

function joinFragment(base, frag) {
  const f = frag.trim();
  if (!f) return base;
  if (!base) return f;
  if (NO_SPACE_BEFORE.test(f)) return base + f;      // "well" + "," -> "well,"
  if (base.endsWith("(")) return base + f;           // "(this" + ")" -> "(this)"
  return base + " " + f;
}

function mergeCaption(lines, who, text) {
  const out = lines.map((l) => ({ who: l.who, text: l.text, open: l.open }));
  if (text == null || !String(text).trim()) return out;
  const last = out[out.length - 1];
  if (last && last.open && last.who === who) {
    last.text = joinFragment(last.text, String(text));
  } else {
    out.push({ who: who, text: String(text).trim(), open: true });
  }
  return out;
}

function endTurn(lines) {
  return lines.map((l) => ({ who: l.who, text: l.text, open: false }));
}

/* ---- orb parameters --------------------------------------------------------
 *
 * Deterministic: same (state, level, t) -> same result, so it is testable
 * under node. The page renders with these plus its own time-based motion.
 *   state:     one of STATES (unknown falls back to "idle")
 *   level:     live audio level 0..1 (mic while listening, playback while speaking)
 *   t:         seconds (any monotonic clock)
 * Returns {color, radius, spin, particles}:
 *   radius     normalised core radius, grows with level, always > 0
 *   spin       orbit speed (rad/s), grows gently with level
 *   particles  how many orbiting particles the page should draw (>= 60)
 */

function orbParams(state, level, t) {
  const style = STATE_STYLE[state] || STATE_STYLE.idle;
  const lv = Math.max(0, Math.min(1, Number(level) || 0));
  const time = Number(t) || 0;
  const breath = Math.sin(time * 1.4) * 0.5 + 0.5; // 0..1 slow breathing
  const radius = 0.16 + lv * 0.14 + breath * style.pulse * 0.05;
  const spin = style.speed * (0.9 + lv * 0.5);
  const particles = 64 + Math.round(lv * 24);
  return { color: style.color, radius: radius, spin: spin, particles: particles };
}

/* ---- sound cues (M9) -------------------------------------------------------
 *
 * Short synthesized status sounds, OUR OWN design (no samples, no files).
 * The page (live.html) synthesizes each name with WebAudio; cueFor maps an
 * incoming server event to one cue name, or null when the event is silent.
 * "listen" is never returned by cueFor: the page plays it when the mic starts.
 */

const CUES = ["listen", "whoosh", "chirp", "chime", "attention", "stop", "error"];

function cueFor(event) {
  if (!event || typeof event !== "object") return null;
  switch (event.type) {
    case "tool_started":
      return "whoosh";                                   // work begins
    case "tool_result":
      if (event.result && typeof event.result === "object") {
        return event.result.ok === false ? "error" : "chirp";
      }
      return "chirp";
    case "hermes_done":
      return "chime";                                    // Hermes finished
    case "approval_needed":
      return "attention";                                // needs the user's yes/no
    case "state":
      return event.state === "muted" ? "stop" : null;
    case "error":
      return "error";
    default:
      return null;
  }
}

/* ---- ambient music ducking -------------------------------------------------
 *
 * Background pad volume: full when nobody is talking, ~15% while 7 or the
 * user speaks, zero while muted. base is the user's slider value (0..1,
 * clamped). Deterministic; testable under node.
 */

const DUCK_FACTOR = 0.15;

function duckVolume(state, base, userSpeaking) {
  let b = Number(base);
  if (!isFinite(b)) b = 0;
  b = Math.max(0, Math.min(1, b));              // clamp to [0, 1]
  if (state === "muted") return 0;
  if (state === "speaking" || userSpeaking) return b * DUCK_FACTOR;
  return b;
}

/* ---- exports (browser global + CommonJS for node) -------------------------- */

const liveCore = { STATES: STATES, STATE_STYLE: STATE_STYLE,
  STATE_LABEL: STATE_LABEL, STATE_WORD: STATE_WORD, labelFor: labelFor,
  mergeCaption: mergeCaption, endTurn: endTurn, orbParams: orbParams,
  duckVolume: duckVolume, CUES: CUES, cueFor: cueFor };

if (typeof module !== "undefined" && module.exports) {
  module.exports = liveCore;
}
if (typeof globalThis !== "undefined") {
  globalThis.liveCore = liveCore;
}
})();
