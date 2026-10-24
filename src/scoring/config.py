"""Scoring configuration constants for Laya delivery rubric.

All scoring constants are kept separate from detection thresholds and configs.
No long dashes in any text or comments.
"""

RUBRIC_VERSION: str = "1.0.0"

# Method formula description as plain text
METHOD_FORMULA_TEXT: str = (
    "Pace and Pitch variation scores = 100 * (1 - flagged_seconds / duration), "
    "where overlapping flagged intervals are merged and clipped to audio duration. "
    "Pauses score = 100 * (1 - flagged_pauses / pauses_examined). "
    "Loudness is measured but unscored. "
    "Overall score is the unweighted arithmetic mean of scored dimensions "
    "(minimum 2 scored dimensions required, otherwise null)."
)

METHOD_NOTE: str = (
    "A score of 100 means the detectors flagged nothing. "
    "It does not prove the delivery was good."
)

RELIABILITY_NOTE_PAIRED: str = "Compared with an ideal recording of the same speech"
RELIABILITY_NOTE_EXPERIMENTAL: str = "Compared with a general baseline, so less reliable"

# One-sentence criteria describing what each dimension checks
CRITERION_PACE: str = (
    "Evaluates speech tempo consistency and flags intervals where "
    "speaking rate is significantly too fast or too slow."
)

CRITERION_PITCH: str = (
    "Evaluates melodic intonation variation and flags intervals where "
    "vocal pitch is abnormally flat or monotone."
)

CRITERION_PAUSES: str = (
    "Evaluates conversational phrasing and flags silent gaps between "
    "words that are abnormally rushed or prolonged."
)

CRITERION_LOUDNESS: str = (
    "Evaluates volume stability and energy dynamics across spoken phrases."
)

REASON_LOUDNESS_UNSCORED: str = (
    "Loudness scoring is not yet calibrated; energy dynamics are "
    "measured and displayed without affecting the delivery score."
)

REASON_PAUSES_UNSCORED: str = (
    "The pause detector examines natural baseline phrase boundaries "
    "(>= 150 ms) and test gaps >= 150 ms, but the examined boundary "
    "count is not preserved in analysis results and cannot be determined "
    "without changing the detector."
)

MIN_SCORED_DIMENSIONS: int = 2
