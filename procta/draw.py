import cv2


def annotate(frame, dets: dict, signals: dict = None, banner_lines=()):
    img = frame.copy()
    for p in dets["persons"]:
        x1, y1, x2, y2 = map(int, p["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"ID {p['id']}", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        for x, y, c in p["kps"]:
            if c > 0.3:
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), -1)
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
