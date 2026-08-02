"""영상 입출력 + 배경 plate 추출 (정적 카메라 가정)."""
import cv2
import numpy as np


def read_frames(path, max_frames=160, stride=None):
    """영상에서 최대 max_frames개 균등 샘플. (frames[list HxWx3 BGR], total_n, fps)"""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        # 일부 컨테이너는 frame count 미보고 -> 순차 디코드
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if stride is None and len(frames) > max_frames:
            idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
            frames = [frames[i] for i in idx]
        return frames, len(frames), fps
    if stride is not None:
        idx = np.arange(0, total, stride)
    else:
        k = min(max_frames, total)
        idx = np.linspace(0, total - 1, k).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames, total, fps


def background_plate(frames, sample=60):
    """프레임 중앙값 = 빈 컨베이어 배경 plate (이동 박스 제거)."""
    if len(frames) > sample:
        idx = np.linspace(0, len(frames) - 1, sample).astype(int)
        stack = np.stack([frames[i] for i in idx])
    else:
        stack = np.stack(frames)
    return np.median(stack.astype(np.float32), axis=0).astype(np.uint8)


def foreground_mask(frame, bg, thr=40, open_k=5, close_k=9, min_area=600):
    """배경차분 전경 마스크 (uint8 0/255) + 큰 컴포넌트 stats."""
    d = np.abs(frame.astype(np.int16) - bg.astype(np.int16)).sum(2)
    m = (d > thr).astype(np.uint8) * 255
    if open_k:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    if close_k:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(m)
    comps = [(stats[i], cent[i]) for i in range(1, n) if stats[i, 4] >= min_area]
    return m, comps
