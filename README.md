# Voice Assistant

A local-first voice assistant CLI. Speech-to-text and text-to-speech run entirely on your machine (Whisper and Piper), and only the transcribed text is sent to an OpenAI-compatible LLM API.

![Python](https://img.shields.io/badge/python-3.12-blue)
![uv](https://img.shields.io/badge/managed%20with-uv-purple)
![License](https://img.shields.io/badge/license-MIT-green)
![STT](https://img.shields.io/badge/STT-Whisper%20(local)-orange)
![TTS](https://img.shields.io/badge/TTS-Piper%20(local)-orange)

## Demo

[Watch the demo](assets/demo.mp4)

Local Whisper STT + Piper TTS: only the transcript leaves your machine.

```bash
uv run main.py
```

## Features

- Push-to-talk recording from the terminal (Enter to start, Enter to stop)
- Local speech-to-text with Whisper, including automatic language detection (French/English)
- Local text-to-speech with Piper, one voice per language
- Multi-turn conversation with history
- OpenAI-compatible LLM backend (any `base_url` + `api_key`)
- Clear error handling: missing env vars, missing voice files, microphone/playback errors, API failures
- Anti-hallucination guard: recordings shorter than 0.5 s are ignored

## Architecture

```
 microphone
     |
     v
 [mic.py]  record until Enter (in-memory, callback + queue)
     |
     v
 [stt.py]  resample to 16 kHz -> Whisper (local) -> detected language + text
     |
     v
 [llm.py]  system prompt + conversation history -> OpenAI-compatible API
     |
     v
 [tts.py]  Piper voice selected from the detected language -> speakers
```

Only the transcribed text leaves your machine. Audio never does.

## Tech stack

| Component | Library / Model |
|---|---|
| Recording and playback | `sounddevice` (PortAudio) |
| Speech-to-text | `transformers` + `openai/whisper-small` (local) |
| Text-to-speech | `piper-tts` + `rhasspy/piper-voices` (local) |
| LLM | `openai` Python SDK against any OpenAI-compatible endpoint |
| Audio resampling | `scipy.signal.resample_poly` |
| Environment | Python 3.12, `uv` |

## Quickstart

Prerequisites: Python 3.12, [`uv`](https://docs.astral.sh/uv/), a working microphone, and an API key for an OpenAI-compatible LLM endpoint.

```bash
git clone https://github.com/afrahsanaa/voice-assistant.git
cd voice-assistant
uv sync
```

Create your `.env` from the template:

```bash
cp .env.example .env
```

On Windows:

```powershell
Copy-Item .env.example .env
```

```dotenv
BASE_URL=https://your-llm-endpoint/v1
API_KEY=your-api-key
LLM_MODEL=your-model-name
WHISPER_MODEL=openai/whisper-small
```

Download the Piper voices (one per supported language):

```bash
uv run python -m piper.download_voices --download-dir voices fr_FR-siwis-medium en_US-lessac-medium
```

Run it:

```bash
uv run main.py
```

## Usage

```
==============================================
 Voice Assistant | local Whisper + Piper
==============================================
Loading models...
Models ready in 8.2s.

Ready. Say "quit" or "exit", or press Ctrl+C to stop.

Press Enter to start recording...
Recording... press Enter to stop.
Transcribing...
You [fr]: Quelle est la capitale de la France ?
Thinking...
Assistant: La capitale de la France est Paris.
Speaking...

Goodbye.
```

- Say `quit` or `exit`, or press Ctrl+C, to stop.
- Recordings shorter than 0.5 s are skipped to avoid Whisper hallucinations on silence.
- If the LLM request fails, the turn is rolled back and the assistant asks again.

### Standalone modules

Each module is also a small CLI on its own, useful for debugging a single stage:

```bash
uv run mic.py                          # record and save a WAV in recordings/
uv run stt.py recordings/recording.wav # transcribe a file
uv run llm.py                          # text-only chat
uv run tts.py "Bonjour tout le monde"  # speak text (--voice to pick another voice)
```

## Project structure

| File | Purpose |
|---|---|
| `main.py` | CLI loop: wires recording, STT, LLM and TTS together |
| `mic.py` | `record()`: push-to-talk capture with a callback and a queue |
| `stt.py` | `load_whisper()`, `prepare_audio()`, `transcribe()` |
| `llm.py` | `create_client()`, `ask_llm()`, system prompt, history |
| `tts.py` | `load_voice()`, `speak()` (Piper) |
| `assets/demo.mp4` | Terminal demo recording |
| `.env.example` | Required environment variables |
| `pyproject.toml`, `uv.lock` | Dependencies (managed with `uv`) |

## Design decisions

- **Audio in memory.** Recordings are accumulated in a queue and concatenated as a NumPy array; no WAV file is written during normal use. The callback only copies buffers, so the audio thread never does heavy work.
- **One detection pass.** The language is detected once with `detect_language` and reused in `generate(language=...)`, instead of letting the model detect it again.
- **Sentence-level playback.** Piper synthesizes one chunk per sentence, and each chunk is played as soon as it is ready, which keeps the perceived latency low.
- **Speech-safe answers.** The system prompt forbids markdown, lists and emojis, and `clean_text_for_speech()` in `tts.py` strips any formatting that still slips through before synthesis.
- **Language-aware voices.** The detected language token (`<|fr|>` / `<|en|>`) selects the Piper voice, with English as fallback.
- **Recoverable failures.** API errors pop the pending user message so the conversation history stays consistent, and the loop continues instead of crashing.

## Limitations and roadmap

- Whisper runs on CPU: expect a few seconds per utterance with `whisper-small`. Use `openai/whisper-base` or `openai/whisper-tiny` for lower latency.
- No streaming STT/TTS yet; each turn is a full request.
- Conversation history is in memory and unbounded.
- Roadmap: web/PWA version for phone use, streaming transcription, voice activity detection, multiple assistant personas.

## Privacy

- Speech-to-text and text-to-speech are fully local.
- Only the transcribed text is sent to the LLM endpoint you configure.
- `.env`, `recordings/` and `voices/` are git-ignored.

## License

MIT
