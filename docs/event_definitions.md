# Event definitions (v1)

An **alert** is a prompt for a human invigilator to look, never a judgement that a student cheated.
An event only produces an alert if it persists for **≥ 3 s** (meeting-notes rule). Each definition
below says what counts, what must **not** trigger an alert, and how the label is marked in ground truth.

| Event | Counts as the event | Must NOT trigger an alert | GT label start / end |
|---|---|---|---|
| `phone_in_hand` | A mobile phone is visible and held in a hand of the seated student (using it, looking at it, holding it low near the desk edge) | Phone glanced at or touched for < 3 s; a calculator, pencil case, eraser or ruler in hand; a phone in someone else's hand | First frame the phone is in the hand → last frame before it leaves the hand |
| `phone_on_desk` | A phone is visible on the student's own desk, not in the hand | Phone on the invigilator's desk or a neighbour's desk; phone in a closed bag; phone already counted as in-hand | Phone placed on desk → phone removed |
| `body_movement` | Sustained large upper-body movement: repeatedly turning around, looking behind, twisting towards a neighbour | Stretching, adjusting posture, reaching for a pen, a single head turn, < 3 s of movement | First frame of the sustained movement → last |
| `unusual_seat_movement` | Student's body leaves their normal seat position: leaning far into the aisle/neighbour's space, sliding the chair, standing up | Leaning forward to write, small posture shifts, sitting back | Leaves the normal position → returns |
| `head_turned` | Head turned clearly left or right away from the student's own work/screen (e.g. towards a neighbour's paper) and held | A head turn or glance < 3 s; looking down at own paper (normal when writing; see `looking_down`); small head tilts | Head turns away → faces forward again |
| `looking_down` | Head lowered further than the student's own writing pose AND hands off the desk (lap / under the desk / not visible), held | Writing or reading one's own paper (head down, hands on the desk); picking up a pen < 3 s | Head drops below the writing pose with hands off the desk → back |
| `material_exchange` | Hands of two different students meet between their desks (same row or front ↔ back), with or without a visible sheet; ≥ 1 s | Handing paper to/from the invigilator (authorised); stretching; hands near each other while writing | Hands within reach between the desks → separate |

## Known ambiguities to discuss
- `body_movement` overlaps with `unusual_seat_movement` (leaning sideways is both). For v1 they are labelled independently; overlapping labels are allowed.
- `phone_on_desk` may be permitted in some exam rules (phone face-down, switched off). It is kept as a low-priority event whose relevance the invigilator decides.
- `head_turned` overlaps with `body_movement` when turning around; the protocol labels both (`body_movement+head_turned`).
  It is measured from face keypoints (nose vs eyes) relative to the yaw recorded during seat calibration, so the student must face their work during the first ~2 s.
- Track identity: staged sessions use one student per camera so evaluation ignores track ids; multi-student scenes are a later step.
