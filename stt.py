import os
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_VERBOSITY', 'error')

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

SAMPLE_RATE = 16000


def load_whisper(model_name):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    return processor, model


def prepare_audio(audio, samplerate):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if samplerate != SAMPLE_RATE:
        gcd = np.gcd(samplerate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // gcd, samplerate // gcd)
    return audio


def transcribe(audio, processor, model):
    input_features = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors='pt',
    ).input_features
    language = processor.tokenizer.convert_ids_to_tokens(
        model.detect_language(input_features)
    )[0]
    predicted_ids = model.generate(input_features, language=language)
    text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return language, text


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: uv run stt.py <audio file>')

    load_dotenv()
    model_name = os.getenv('WHISPER_MODEL')
    if not model_name:
        raise SystemExit('Missing required environment variable: WHISPER_MODEL')

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        raise FileNotFoundError(f'Missing audio file: {audio_path.resolve()}')

    audio, samplerate = sf.read(audio_path, dtype='float32')
    audio = prepare_audio(audio, samplerate)

    processor, model = load_whisper(model_name)
    language, text = transcribe(audio, processor, model)

    print(f'Language: {language.strip("<|>")}')
    print(f'Text: {text}')


if __name__ == '__main__':
    main()
