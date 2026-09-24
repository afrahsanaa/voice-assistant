import argparse
import queue
import sys
from datetime import datetime
from pathlib import Path

import numpy
import sounddevice as sd
import soundfile as sf

assert numpy  


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


def record(device=None, channels=1, samplerate=None):
    """Record from the microphone until Enter is pressed.

    Returns (audio, samplerate). audio is None if the user cancelled with Ctrl+C.
    """
    if samplerate is None:
        device_info = sd.query_devices(device, 'input')
        samplerate = int(device_info['default_samplerate'])

    q = queue.Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(f'Audio warning: {status}', file=sys.stderr)
        q.put(indata.copy())

    print('Press Enter to start recording...')
    try:
        input()
    except KeyboardInterrupt:
        print('\nCancelled before recording started.')
        return None, samplerate

    try:
        with sd.InputStream(
            samplerate=samplerate,
            device=device,
            channels=channels,
            callback=callback,
        ):
            print('Recording... press Enter to stop.')
            input()
    except KeyboardInterrupt:
        print('\nInterrupted during recording.')
        return None, samplerate

    audio_chunks = []
    while True:
        try:
            audio_chunks.append(q.get_nowait())
        except queue.Empty:
            break

    if not audio_chunks:
        return numpy.zeros((0, channels), dtype='float32'), samplerate

    return numpy.concatenate(audio_chunks), samplerate


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    parser.add_argument(
        'filename', nargs='?', metavar='FILENAME',
        help='audio file to store recording to')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-r', '--samplerate', type=int, help='sampling rate')
    parser.add_argument(
        '-c', '--channels', type=int, default=1, help='number of input channels')
    parser.add_argument(
        '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    output_dir = Path('recordings')
    output_dir.mkdir(exist_ok=True)

    if args.filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = output_dir / f'recording_{timestamp}.wav'
    else:
        filename = Path(args.filename)
        if not filename.is_absolute():
            filename = output_dir / filename.name
        filename.parent.mkdir(parents=True, exist_ok=True)

    try:
        audio, samplerate = record(
            device=args.device,
            channels=args.channels,
            samplerate=args.samplerate,
        )
    except KeyboardInterrupt:
        print('\nCancelled.')
        return

    if audio is None:
        return

    if len(audio) == 0:
        print(f'Warning: no audio frames were captured; {filename} was not written.')
        return

    with sf.SoundFile(
        str(filename),
        mode='x',
        samplerate=samplerate,
        channels=args.channels,
        subtype=args.subtype,
    ) as file:
        file.write(audio)

    print(f'Saved: {filename}')


if __name__ == '__main__':
    main()
