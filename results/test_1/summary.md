# Session `test_1` - all methods, observation point T1

## Object detector: `world`  (4.5 min, 4066 analysed frames, stride 2, extraction 1454.7 s)

| method | alerts | per minute | phone_in_hand | phone_on_desk | body_movement | unusual_seat_movement | head_turned | looking_down | material_exchange |
|---|---|---|---|---|---|---|---|---|---|
| frame | 908 | 201.0 | 32 | 153 | 264 | 20 | 352 | 42 | 45 |
| baseline | 63 | 13.9 | 2 | 10 | 0 | 1 | 42 | 4 | 4 |
| improved | 26 | 5.8 | 1 | 0 | 0 | 1 | 14 | 6 | 4 |
| context | 26 | 5.8 | 1 | 0 | 0 | 1 | 14 | 6 | 4 |

Alerts per seat (context mode):

| seat | phone_in_hand | phone_on_desk | body_movement | unusual_seat_movement | head_turned | looking_down | material_exchange |
|---|---|---|---|---|---|---|---|
| B1 | 0 | 0 | 0 | 0 | 3 | 1 | 0 |
| B1+B3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| B1+F1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| B2 | 1 | 0 | 0 | 0 | 1 | 3 | 0 |
| B2+B3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| B3 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| B3+F2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F1 | 0 | 0 | 0 | 0 | 3 | 1 | 0 |
| F1+F2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| F2 | 0 | 0 | 0 | 0 | 2 | 1 | 0 |
| F2+F3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F3 | 0 | 0 | 0 | 1 | 3 | 0 | 0 |

Logged instead of alerted (context mode, frame counts): {'calculator in hand (permitted device)': 11492}

## Object detector: `coco`  (4.5 min, 4066 analysed frames, stride 2, extraction 1183.4 s)

| method | alerts | per minute | phone_in_hand | phone_on_desk | body_movement | unusual_seat_movement | head_turned | looking_down | material_exchange |
|---|---|---|---|---|---|---|---|---|---|
| frame | 1675 | 370.8 | 294 | 658 | 264 | 20 | 352 | 42 | 45 |
| baseline | 107 | 23.7 | 17 | 39 | 0 | 1 | 42 | 4 | 4 |
| improved | 29 | 6.4 | 3 | 1 | 0 | 1 | 14 | 6 | 4 |
| context | 29 | 6.4 | 3 | 1 | 0 | 1 | 14 | 6 | 4 |

Alerts per seat (context mode):

| seat | phone_in_hand | phone_on_desk | body_movement | unusual_seat_movement | head_turned | looking_down | material_exchange |
|---|---|---|---|---|---|---|---|
| B1 | 0 | 0 | 0 | 0 | 3 | 1 | 0 |
| B1+B3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| B1+F1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| B2 | 1 | 0 | 0 | 0 | 1 | 3 | 0 |
| B2+B3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| B3 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| B3+F2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F1 | 0 | 0 | 0 | 0 | 3 | 1 | 0 |
| F1+F2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| F2 | 1 | 0 | 0 | 0 | 2 | 1 | 0 |
| F2+F3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| F3 | 1 | 1 | 0 | 1 | 3 | 0 | 0 |

Logged instead of alerted (context mode, frame counts): none
