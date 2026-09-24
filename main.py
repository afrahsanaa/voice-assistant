import time
from pathlib import Path

import sounddevice as sd

from llm import ask_llm, create_client, new_messages, required_env
from mic import record
from stt import load_whisper, prepare_audio, transcribe
from tts import load_voice, speak

SAMPLE_RATE = 16000
MIN_DURATION_S = 0.5
VOICES = {
    'fr': Path('voices/fr_FR-siwis-medium.onnx'),
    'en': Path('voices/en_US-lessac-medium.onnx'),
}
DEFAULT_LANGUAGE = 'en'


def load_models(whisper_model_name):
    print('Loading models...')
    started = time.perf_counter()
    processor, model = load_whisper(whisper_model_name)
    voices = {language: load_voice(path) for language, path in VOICES.items()}
    print(f'Models ready in {time.perf_counter() - started:.1f}s.')
    return processor, model, voices


def main():
    print('=' * 46)
    print(' Voice Assistant | local Whisper + Piper')
    print('=' * 46)
    client, llm_model = create_client()
    processor, whisper_model, voices = load_models(required_env('WHISPER_MODEL'))
    messages = new_messages()

    print()
    print('Ready. Say "quit" or "exit", or press Ctrl+C to stop.')

    while True:
        print()
        try:
            audio, samplerate = record()
        except sd.PortAudioError as error:
            raise SystemExit(f'Microphone error: {error}')

        if audio is None:
            break

        audio = prepare_audio(audio, samplerate)
        if len(audio) < MIN_DURATION_S * SAMPLE_RATE:
            print('Recording too short, try again.')
            continue

        print('Transcribing...')
        language, text = transcribe(audio, processor, whisper_model)
        if not text:
            print('Nothing was transcribed, try again.')
            continue

        code = language.strip('<|>')
        print(f'You [{code}]: {text}')

        if text.lower().strip(' .!?') in {'quit', 'exit'}:
            break

        print('Thinking...')
        answer = ask_llm(client, llm_model, messages, text)
        if answer is None:
            continue
        print(f'Assistant: {answer}')

        print('Speaking...')
        try:
            speak(answer, voices.get(code, voices[DEFAULT_LANGUAGE]))
        except sd.PortAudioError as error:
            print(f'Playback error: {error}')

    print()
    print('Goodbye.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nGoodbye.')
