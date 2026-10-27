"""Run the live estimator + policy on a recording (the human's touches act as the sent input).

Outputs <out>/<name>.mp4 (frame | estimated ego grid with policy vs human) and
<out>/<name>.json with per-frame policy probability and agreement statistics.
The policy does not drive here: the car follows the human, so after the first
disagreement the situation differs from what the policy would have caused.

Usage: python3 tools/replay_estimator.py runs/ppo-002 data/raw/video_20260130_173426.mp4 [out_dir]
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kart.estimator import Estimator  # noqa: E402
from kart.ppo import ActorCritic, load_params  # noqa: E402
from kart.touch import clean, detect_touch  # noqa: E402
from kart.vecenv import G  # noqa: E402
from kart.video import unique_frames  # noqa: E402

RELEASE_SHIFT = 0.125  # the recording's touch marker lingers after the real release


def grid_image(g, size=360):
    # same colours as the page: road blue, void pale, unknown grey; rotated like the game camera
    lut = {1.0: (189, 111, 47), 0.0: (242, 230, 207), 0.5: (194, 176, 159)}
    img = np.zeros((G, G, 3), np.uint8)
    for k, c in lut.items():
        img[g == k] = c
    big = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    # u (columns) -> up-right, v (rows) -> up-left: the square becomes a diamond
    h = size / 2
    src = np.float32([[0, 0], [size, 0], [0, size]])
    dst = np.float32([[h, size], [size, h], [0, h]])
    return cv2.warpAffine(big, cv2.getAffineTransform(src, dst), (size, size), borderValue=(250, 246, 240))


def main(run, video, out_dir="artifacts/viz/replays"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(video).stem
    params = load_params()
    net = ActorCritic()
    net.load_state_dict(torch.load(Path(run) / "best.pt", map_location="cpu"))
    net.eval()
    frames = list(unique_frames(video))
    t = np.array([f[1] for f in frames])
    raw = np.array([detect_touch(f[2]) is not None for f in frames])
    touch = clean(raw)
    # human input actually sent: marker on, minus its fade after release
    sent = np.array([touch[i] and touch[min(np.searchsorted(t, t[i] + RELEASE_SHIFT), len(t) - 1)]
                     for i in range(len(t))])
    est = Estimator(params)
    writer = None
    rows = []
    started = False
    for i, (_, ti, f) in enumerate(frames):
        # the race starts when the car first moves; skip the countdown
        est.update(f, ti, bool(sent[i]))
        if not started:
            started = est.ok_frames > 5 and est.speed() > 600
            if not started:
                continue
        grid, state = est.observe()
        with torch.no_grad():
            logits, _ = net(grid, state)
        p_press = float(torch.softmax(logits, 1)[0, 1])
        rows.append({"t": round(float(ti), 3), "human": int(sent[i]), "p_press": round(p_press, 3),
                     "heading": round(est.heading() * 180 / np.pi, 1), "speed": round(est.speed(), 0),
                     "boost": int(state[0, 5] > 0)})
        left = cv2.resize(f, (360, 800))
        right = np.full((800, 360, 3), 250, np.uint8)
        right[40:400] = grid_image(grid[0, 0].numpy())
        txt = [("t %.2fs" % ti, (40, 40, 40)),
               ("human  %s" % ("PRESS B" if sent[i] else "release A"), (63, 96, 224) if sent[i] else (127, 144, 18)),
               ("policy %s  p=%.2f" % ("PRESS B" if p_press > 0.5 else "release A", p_press),
                (63, 96, 224) if p_press > 0.5 else (127, 144, 18)),
               ("heading %3.0f deg  speed %4.0f" % (est.heading() * 180 / np.pi, est.speed()), (40, 40, 40)),
               ("boost" if state[0, 5] > 0 else "", (0, 154, 217))]
        for k, (s, c) in enumerate(txt):
            cv2.putText(right, s, (14, 450 + 36 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.62, c, 2, cv2.LINE_AA)
        bar = int(330 * p_press)
        cv2.rectangle(right, (14, 650), (344, 672), (220, 220, 220), -1)
        cv2.rectangle(right, (14, 650), (14 + bar, 672), (63, 96, 224), -1)
        cv2.line(right, (179, 645), (179, 677), (40, 40, 40), 1)
        cv2.putText(right, "policy P(press)", (14, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
        frame_out = np.hstack([left, right])
        if writer is None:
            writer = cv2.VideoWriter(str(out / f"{name}.raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                     (frame_out.shape[1], frame_out.shape[0]))
        writer.write(frame_out)
    writer.release()
    # browsers need H.264; OpenCV writes mp4v
    raw = out / f"{name}.raw.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "26", "-preset", "veryfast", "-movflags", "+faststart", str(out / f"{name}.mp4")],
                   check=True)
    raw.unlink()
    h = np.array([r["human"] for r in rows])
    pp = np.array([r["p_press"] for r in rows]) > 0.5
    tt = np.array([r["t"] for r in rows])
    # switch timing: for each human switch, the nearest policy switch of the same kind within 0.5 s
    hs = np.nonzero(np.diff(h))[0] + 1
    ps = np.nonzero(np.diff(pp.astype(int)))[0] + 1
    lags = []
    for k in hs:
        same = [j for j in ps if pp[j] == bool(h[k]) and abs(tt[j] - tt[k]) < 0.5]
        if same:
            j = min(same, key=lambda j: abs(tt[j] - tt[k]))
            lags.append(tt[j] - tt[k])
    summary = {"video": name, "run": str(run), "frames": len(rows), "agreement": float((h == pp).mean()),
               "human_switches": int(len(hs)), "policy_switches": int(len(ps)),
               "matched_switches": len(lags),
               "policy_minus_human_s": {"median": float(np.median(lags)) if lags else None,
                                        "p25": float(np.percentile(lags, 25)) if lags else None,
                                        "p75": float(np.percentile(lags, 75)) if lags else None}}
    (out / f"{name}.json").write_text(json.dumps({"summary": summary, "rows": rows}))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
