import cv2

# COCO-17 skeleton: face (nose, eyes, ears), arms, torso, legs
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4),            # nose-eyes-ears
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # shoulders, arms
            (5, 11), (6, 12), (11, 12),                # torso
            (11, 13), (13, 15), (12, 14), (14, 16)]    # legs
FACE = {0, 1, 2, 3, 4}
KP_MIN_CONF = 0.3


def draw_pose(img, kps):
    pts = [(int(x), int(y)) if c > KP_MIN_CONF else None for x, y, c in kps]
    for a, b in SKELETON:
        if pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], (255, 180, 0), 2)
    for i, p in enumerate(pts):
        if p:
            color = (0, 255, 255) if i in FACE else (0, 140, 255)   # eyes/nose/ears yellow
            cv2.circle(img, p, 4, color, -1)
            cv2.circle(img, p, 4, (0, 0, 0), 1)


def annotate(frame, dets: dict, signals: dict = None, banner_lines=(), show_pose=True):
    img = frame.copy()
    for p in dets["persons"]:
        x1, y1, x2, y2 = map(int, p["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"ID {p['id']}", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        if show_pose:
            draw_pose(img, p["kps"])
    for ph in dets["phones"]:
        x1, y1, x2, y2 = map(int, ph["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"phone {ph['conf']:.2f}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    y = 20
    if signals:
        for (tid, ev), conf in signals.items():
            cv2.putText(img, f"track {tid}: {ev} ({conf:.2f})", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            y += 18
    for line in banner_lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        y += 20
    return img
