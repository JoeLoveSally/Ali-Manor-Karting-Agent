"""Render recorded policy traces into a local interactive gallery and MP4s."""
import argparse
import base64
import json
import math
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kart.scenes import sha256

BG, ROAD, EDGE = '#0e1726', '#3d5873', '#7992aa'
VOID, A, B, AIR, OK, BAD = '#08101b', '#36d6b3', '#ff9477', '#80caff', '#6de0a0', '#ff7885'
INK, MUTED = '#f0f5fc', '#a7b7cc'
W, H, FPS, SPEED = 1280, 800, 20, 2.


@lru_cache(None)
def font(size):
    path = os.environ.get('KART_FONT', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    return ImageFont.truetype(path, size)


def text(draw, xy, value, size=22, color=INK):
    draw.text(xy, str(value), font=font(size), fill=color)


def mapper(case, size, row=None):
    width, height = size
    if row is None:
        rects = np.asarray(case['rects'])
        # Batched local scenes pad unused rectangles with 1e9 sentinels.
        rects = rects[np.isfinite(rects).all(1) & (np.abs(rects) < 1e8).all(1)
                      & (rects[:, 2:] > rects[:, :2]).all(1)]
        if not len(rects):
            raise ValueError('Replay has no valid road rectangles')
        lo, hi = rects[:, :2].min(0)-100, rects[:, 2:].max(0)+100
        scale = min((width-36)/(hi[0]-lo[0]), (height-36)/(hi[1]-lo[1]))
        center = (lo+hi)/2
    else:
        scale = min(width, height)/2800
        center = np.array([row[1]+450, row[2]+450])
    return lambda x, y: (width/2+(x-center[0])*scale, height/2-(y-center[1])*scale)


def map_image(case, size, index, follow=False):
    row = case['frames'][index]
    image = Image.new('RGB', size, VOID)
    draw = ImageDraw.Draw(image)
    point = mapper(case, size, row if follow else None)
    def rect(r, fill, outline=None):
        x0, y0 = point(r[0], r[3]); x1, y1 = point(r[2], r[1])
        if x1 >= 0 and y1 >= 0 and x0 < size[0] and y0 < size[1]:
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=outline, width=1)
    for r in case['rects']:
        rect(r, ROAD, EDGE)
    for axis, start, end, l0, l1, height in case.get('reliefs', []):
        for k in range(16):
            a, b = start+(end-start)*k/16, start+(end-start)*(k+1)/16
            weight = math.sin(math.pi*(k+.5)/16)**2
            base = (61, 88, 115)
            peak = (154, 115, 63) if height > 0 else (66, 120, 151)
            color = tuple(round(x+(y-x)*weight) for x, y in zip(base, peak))
            rect([a, l0, b, l1] if axis == 0 else [l0, a, l1, b], color)
    for axis, start, lip, l0, l1, _ in case['ramps']:
        rect([start, l0, lip, l1] if axis == 0 else [l0, start, l1, lip], '#966b37', '#e9b16c')
    for r in case['holes']:
        rect(r, VOID)
    for axis, lip, land, l0, l1, _ in case['jumps']:
        for along, color in [(lip, '#ffd382'), (land, AIR)]:
            ends = [(along, l0), (along, l1)] if axis == 0 else [(l0, along), (l1, along)]
            draw.line([point(*p) for p in ends], fill=color, width=2)
    begin = max(1, index-180) if follow else 1
    for k in range(begin, index+1):
        p, q = case['frames'][k-1:k+1]
        draw.line([point(p[1], p[2]), point(q[1], q[2])], fill=B if q[6] else A, width=3 if follow else 2)
    for p, color in [(case['frames'][0][1:3], OK), (case['corners'][-1], '#cbd8eb')]:
        x, y = point(*p)
        if -10 <= x <= size[0]+10 and -10 <= y <= size[1]+10:
            draw.ellipse((x-4, y-4, x+4, y+4), fill=color)
    x, y = point(row[1], row[2])
    rad = 13 if follow else 6
    if row[8]:
        draw.ellipse((x-rad-5, y-rad-5, x+rad+5, y+rad+5), outline=AIR, width=3)
    if row[9] > 0:
        draw.ellipse((x-rad-9, y-rad-9, x+rad+9, y+rad+9), outline='#ffd382', width=2)
    theta = row[3]
    direction = np.array([np.sin(theta), -np.cos(theta)])
    side = np.array([np.cos(theta), np.sin(theta)])
    tip = np.array([x, y])+rad*direction*1.6
    p0 = np.array([x, y])-rad*.7*direction+rad*.8*side
    p1 = np.array([x, y])-rad*.7*direction-rad*.8*side
    draw.polygon([tuple(tip), tuple(p0), tuple(p1)], fill=INK, outline=BG)
    if index == len(case['frames'])-1 and not case['finished']:
        draw.line((x-rad, y-rad, x+rad, y+rad), fill=BAD, width=4)
        draw.line((x-rad, y+rad, x+rad, y-rad), fill=BAD, width=4)
    if follow:
        text(draw, (16, 12), 'A ↑   B →', 20, MUTED)
    return image


def compose(case, observations, index):
    row = case['frames'][index]
    image = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (24, 10), f'地图 {case["number"]:02d} · {case["label"]}', 29)
    text(draw, (650, 17), f'{case.get("model_name", "best")} · 合成仿真 · 2 倍速', 23, MUTED)
    text(draw, (24, 64), '跟随视图 · 俯视道路', 23)
    text(draw, (768, 64), '全图与已行驶轨迹', 20)
    text(draw, (1020, 64), '模型实际输入', 20)
    image.paste(map_image(case, (720, 620), index, follow=True), (24, 104))
    image.paste(map_image(case, (232, 340), index), (768, 104))
    oi = int(row[11])
    grid = np.flipud(observations[oi])
    image.paste(Image.fromarray(grid).convert('RGB').resize((232, 232), Image.Resampling.NEAREST), (1020, 104))
    draw = ImageDraw.Draw(image)
    text(draw, (1014, 350), '白：道路  灰：未知', 17, MUTED)
    text(draw, (1014, 377), '黑：空缺 / 非路面', 17, MUTED)
    ended = index == len(case['frames'])-1
    state = ('通过' if case['finished'] else '超过时限' if case.get('timeout') else
             '地图未知' if case.get('unknown') else '落地余量不足' if row[17] else
             '离路失败') if ended else ('腾空' if row[8] else '地面 / 坡面')
    color = OK if ended and case['finished'] else BAD if ended else AIR if row[8] else INK
    text(draw, (768, 466), f'时间  {row[0]:.2f} / {case["seconds"]:.2f} 秒', 23)
    text(draw, (768, 506), '按住屏幕 · B' if row[6] else '松开屏幕 · A', 30, B if row[6] else A)
    text(draw, (768, 558), f'状态  {state}', 25, color)
    text(draw, (768, 603), f'速度 {row[5]:.0f}    航向 {np.degrees(row[3]):.1f}°', 21)
    text(draw, (768, 641), f'高度 {row[4]:.0f}    生效输入 {"B" if row[7] else "A"}', 21, MUTED)
    text(draw, (768, 679), f'起跳 {case["launches"]} 次 · 安全落地 {case["safe_landings"]} 次', 20, MUTED)
    y = 748
    draw.rounded_rectangle((24, y, W-24, y+9), radius=4, fill=ROAD)
    draw.rounded_rectangle((24, y, 24+(W-48)*min(1, row[0]/case['seconds']), y+9), radius=4, fill=A)
    text(draw, (24, 767), '轨迹：绿为松开，橙为按住；棕色为上凸，蓝色为整幅下凹；起伏区域仍是完整路面。', 17, MUTED)
    return image


def export(root):
    root = Path(root)
    data = json.loads((root/'traces.json').read_text())
    observations = np.load(root/'observations.npz')
    all_videos = []
    for case in data['cases']:
        case['model_name'] = data.get('model_name', 'best')
        grids = observations[case['id']]
        cols = 16
        sprite = Image.new('L', (cols*48, math.ceil(len(grids)/cols)*48), 128)
        for index, grid in enumerate(grids):
            sprite.paste(Image.fromarray(np.flipud(grid)), ((index % cols)*48, (index//cols)*48))
        sprite_path = root/f'{case["id"]}-inputs.png'
        sprite.save(sprite_path)
        case['sprite'] = 'data:image/png;base64,'+base64.b64encode(sprite_path.read_bytes()).decode()
        case['sprite_columns'] = cols
        video_path = root/f'{case["id"]}.mp4'
        times = np.asarray(case['frames'])[:, 0]
        length = int(np.ceil(case['seconds']/SPEED*FPS))+FPS
        with av.open(str(video_path), 'w') as target:
            stream = target.add_stream('libx264', rate=FPS)
            stream.width, stream.height, stream.pix_fmt = W, H, 'yuv420p'
            stream.options = {'crf': '24', 'preset': 'veryfast'}
            for k in range(length):
                t = min(case['seconds'], k/FPS*SPEED)
                index = min(len(times)-1, max(0, int(np.searchsorted(times, t, side='right')-1)))
                canvas = compose(case, grids, index)
                frame = av.VideoFrame.from_image(canvas)
                frame.pts, frame.time_base = k, Fraction(1, FPS)
                for packet in stream.encode(frame):
                    target.mux(packet)
            for packet in stream.encode():
                target.mux(packet)
        poster = root/f'{case["id"]}.png'
        compose(case, grids, len(case['frames'])-1).save(poster)
        case['video'], case['poster'] = video_path.name, poster.name
        all_videos.append(video_path.name)
        print(f'Rendered {video_path.name}: {length} frames', flush=True)
    overview = Image.new('RGB', (1800, 1030), BG)
    draw = ImageDraw.Draw(overview)
    text(draw, (28, 12), f'{data.get("model_name", "best")} · {len(data["cases"])} 张地图', 37)
    passed = sum(c['finished'] for c in data['cases'])
    text(draw, (28, 68), f'合成仿真 · {passed} 张通过、{len(data["cases"])-passed} 张失败 · 分层示例，不代表通过率', 22, MUTED)
    for k, case in enumerate(data['cases']):
        x, y = 22+(k % 5)*356, 122+(k//5)*440
        text(draw, (x+8, y), f'{k+1:02d}  {case["label"]}', 25)
        text(draw, (x+8, y+40), f'{"通过" if case["finished"] else "失败"}  ·  {case["seconds"]:.2f} 秒', 21, OK if case['finished'] else BAD)
        overview.paste(map_image(case, (332, 330), len(case['frames'])-1), (x+8, y+82))
    overview.save(root/'overview.png')
    template = Path(__file__).with_name('policy_replays.html').read_text()
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    (root/'index.html').write_text(template.replace('__REPLAY_DATA__', payload))
    concat = root/'video-list.txt'
    concat.write_text(''.join(f"file '{name}'\n" for name in all_videos))
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
    subprocess.run([ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(concat), '-c', 'copy', '-movflags', '+faststart', str(root/'compilation.mp4')], check=True)
    concat.unlink()
    manifest = dict(cases=len(data['cases']), passed=sum(c['finished'] for c in data['cases']),
                    failed=sum(not c['finished'] for c in data['cases']),
                    all_match_prior_evaluation=all(c['evaluation_parity'] for c in data['cases']),
                    model_sha256=data['model_sha256'], files={p.name: sha256(p) for p in sorted(root.iterdir())
                    if p.is_file() and p.name != 'manifest.json'})
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Gallery, overview and compilation ready', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    export(parser.parse_args().directory)
