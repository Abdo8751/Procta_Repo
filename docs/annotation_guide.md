# Annotation guide (v2, Oct 2026)

A label says **what was visible**, never whether it was misconduct. "phone_in_hand" means a phone
was seen in the participant's hand; whether that breaks the exam rules is decided by the exam
context file and, finally, by the invigilator. Annotators never write "cheating".

Ground truth is one CSV per session (`data/labels/<session>_gt.csv`):
`event_type,start_s,end_s,note`. The recording protocol pre-fills it; correct the times on video
with `python scripts/label_video.py --video data/videos/<session>.mp4`.

## Category 1 - possible mobile-phone use (`phone_in_hand`)

| | Definition |
|---|---|
| Observable criteria | A phone is visible and held in the participant's hand (looking at it, typing, holding it low near the desk edge). |
| Start | First frame the phone is in the hand. |
| End | Last frame before the phone leaves the hand. |
| Target event | Held for at least 3 s. |
| Normal lookalikes (`lookalike_*`) | Permitted calculator in hand; phone glanced at or touched < 3 s; pencil case, eraser or ruler in hand. |
| Uncertain / occluded | Phone partly covered by the hand or the desk edge; label it `phone_in_hand` with note `occluded` if a human can still see it is a phone, otherwise leave it unlabelled. |
| Repeated sightings | Phone put down for less than 30 s and taken out again = **one** labelled event spanning both sightings (note `repeat`). The system should produce one record. |

## Category 2 - possible viewing of another participant's paper (`head_turned`)

| | Definition |
|---|---|
| Observable criteria | Head (or head and upper body) turned towards the neighbouring desk and held. Orientation alone does not show that anything was read. |
| Start | Head begins to turn towards the neighbour desk. |
| End | Head faces the participant's own paper again. |
| Target event | Held for at least 3 s, towards the **neighbour desk**. |
| Normal lookalikes (`lookalike_*`) | Looking at the wall clock, at the invigilator, raising a hand to ask a question, a glance < 3 s, looking down at one's own paper. |
| Uncertain / occluded | Face partly covered by a hand, head turned away from the camera; label with note `occluded`. |
| Repeated sightings | Several looks at the neighbour desk with short breaks (< 30 s) = **one** labelled event (note `repeat`). |

## Not events, logged separately

- `insufficient_view`: the participant cannot be seen clearly (back to the camera, out of frame). The
  robot logs it and moves to the next permitted observation point; it is never an alert.

## Room set-up for solo recordings (protocol `docs/protocol_v2_solo.csv`)

1. One desk for the participant, an empty "neighbour" desk with an answer sheet on one side, a clock (or a
   printed clock) on the other side, and the "invigilator" position on the clock side.
2. Run `python scripts/live.py --mode improved`, turn your head towards the neighbour desk and read the
   yaw label: `image left` or `image right`. Write that side as `neighbour_side` for this observation
   point in `configs/exam_context.json`, and the clock / invigilator side in `allowed_look_sides`.
3. Repeat for each observation point (A, B, C): sides can flip when the camera moves.

## Recording plan (single participant)

| Session | Point | Lighting | Split |
|---|---|---|---|
| s01 | A | daylight | dev |
| s02 | B | daylight | dev |
| s03 | C | evening lamp | dev |
| s04 | A | evening lamp | test |
| s05 | B | evening lamp | test |
| s06 | C | daylight, different clothes | test |

Tune thresholds on s01-s03 only, freeze them (`evaluate_sessions.py --freeze`), then run the test
sessions once. With more participants later, split by participant instead of by session.
