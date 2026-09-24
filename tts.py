import argparse
import re
from pathlib import Path

import sounddevice as sd
from piper import PiperVoice

DEFAULT_VOICE = Path('voices/fr_FR-siwis-medium.onnx')

_MARKDOWN_RULES = [
    (re.compile(r'\[([^\]]+)\]\([^)]*\)'), r'\1'),
    (re.compile(r'\*\*(.+?)\*\*'), r'\1'),
    (re.compile(r'__(.+?)__'), r'\1'),
    (re.compile(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)'), r'\1'),
    (re.compile(r'(?<!\w)_(.+?)_(?!\w)'), r'\1'),
    (re.compile(r'`{1,3}(.+?)`{1,3}'), r'\1'),
    (re.compile(r'^\s{0,3}#{1,6}\s*', re.MULTILINE), ''),
    (re.compile(r'^\s*[-*\u2022]\s+', re.MULTILINE), ''),
    (re.compile(r'^\s*\d+[.)]\s+', re.MULTILINE), ''),
    (re.compile(r'^\s*>\s?', re.MULTILINE), ''),
    (re.compile(r'[*`]+'), ''),
    (re.compile('[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]'), ''),
]


def clean_text_for_speech(text):
    for pattern, replacement in _MARKDOWN_RULES:
        text = pattern.sub(replacement, text)
    return re.sub(r'\s+', ' ', text).strip()


def load_voice(path):
    path = Path(path)
    config_path = Path(f'{path}.json')
    if not path.is_file() or not config_path.is_file():
        raise SystemExit(
            f'missing voice files: {path} and {config_path}. '
            'Download it with: uv run python -m piper.download_voices '
            f'--download-dir voices {path.stem}'
        )
    return PiperVoice.load(str(path))


def speak(text, voice):
    text = clean_text_for_speech(text)
    if not text:
        return
    for chunk in voice.synthesize(text):
        sd.play(chunk.audio_float_array, chunk.sample_rate)
        sd.wait()


def main():
    parser = argparse.ArgumentParser(
        description='Synthesize text with a Piper voice and play it aloud.'
    )
    parser.add_argument(
        'text',
        nargs='+',
        help='text to synthesize',
    )
    parser.add_argument(
        '--voice',
        type=Path,
        default=DEFAULT_VOICE,
        help=f'Piper voice .onnx file (default: {DEFAULT_VOICE})',
    )
    args = parser.parse_args()

    voice = load_voice(args.voice)
    speak(' '.join(args.text), voice)


if __name__ == '__main__':
    main()
