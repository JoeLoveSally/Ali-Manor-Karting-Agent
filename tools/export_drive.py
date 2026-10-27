"""Wrap a recorded drive as a playable H.264 MP4 using its actual frame PTS."""
import argparse
import json
from fractions import Fraction
from pathlib import Path

import av


def export(run):
    run = Path(run)
    times = [json.loads(line)['pts'] for line in (run/'frames.jsonl').read_text().splitlines()]
    target = run/'review.mp4'
    count = 0
    with av.open(str(run/'capture.h264')) as source, av.open(str(target), 'w') as output:
        stream = output.add_stream('libx264', rate=60)
        stream.width, stream.height, stream.pix_fmt = 720, 1600, 'yuv420p'
        stream.options = {'crf': '26', 'preset': 'veryfast'}
        stream.time_base = stream.codec_context.time_base = Fraction(1, 90000)
        for i, frame in enumerate(source.decode(video=0)):
            if i >= len(times):
                raise ValueError('More decoded frames than timestamp records')
            frame.pts = round((times[i]-times[0])*90000)
            frame.time_base = Fraction(1, 90000)
            for packet in stream.encode(frame):
                output.mux(packet)
            count += 1
        for packet in stream.encode():
            output.mux(packet)
    if count != len(times):
        raise ValueError(f'Only {count}/{len(times)} frames exported')
    print(json.dumps({'path': str(target), 'frames': count, 'duration': times[-1]-times[0]}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    export(p.parse_args().run)
