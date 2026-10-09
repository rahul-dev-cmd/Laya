   # Pre-registered pause baseline design

   Date: 9 October 2026

   Pause baselines are computed per boundary type (sentence-final, comma, other).
   If a type has fewer than 8 pauses in a speech, fall back to the pooled baseline.
   Flag a pause at |z| >= 2.0.

   To be evaluated only on speeches added after this date. Not tuned on clips 01 and 02.

   Reason: on clip 02, Mode B flagged 4 of 25 clean natural pauses, all sentence-final,
   because comma and sentence-final pauses were pooled.