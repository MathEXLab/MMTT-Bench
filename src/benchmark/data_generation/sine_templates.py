import random
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Word banks
# ---------------------------------------------------------------------------

TEMPLATES = [
    # 0 — adverb-medial:
    "The {signal} {time_w} {verb} {adverb}.",
    # 1 — imperative-like:
    "Expect the {signal} to {verb} {adverb}.",
    # 2 — fronted time clause:
    "{time_clause}, the {signal} {time_w} {verb} {adverb}.",
    # 3 — nominal subject:
    "A {adj} {noun} {time_w} characterise the signal.",
    # 4 — passive perception:
    "The {signal} {time_w} be observed to be {gerund} {adverb}.",
    # 5 — existential:
    "There {time_w} be a {adj} {noun} in the signal.",
    # 6 — nominal predicate:
    "The {signal}'s behaviour {time_w} take a {adj} {noun}.",
    # 7 — participial absolute:
    "{gerund} {adverb}, the {signal} {time_w} continue past this point.",
    # 8 — fronted clause + nominal:
    "{time_clause}, a {adj} {noun} is evident.",
    # 9 — wh-nominal subject:
    "What characterises the {signal} {time_w} be its {adj} {noun}.",
]

SIGNAL_WORDS = [
    "sine wave", "sinusodial signal", "sinusoid", "simple harmonic motion signal",
    "sine curve", "sinusoid waveform", "sinus wave", "flux signal", "pure tone",
    "harmonic wave"
]


TIME_WORDS = {
    "future":  ["will", "is about to", "is expected to",
                "will soon", "is going to"],
}

TIME_CLAUSES = {
    "future":  ["Going forward", "From this point on",
                "In the near term", "Looking ahead"],
}

DIRECTION = {
    "increasing": {
        "verb":    ["rise", "increase", "climb", "ascend", "grow"],
        "noun":    ["rise", "increase", "ascent", "climb", "upward movement"],
        "gerund":  ["rising", "increasing", "climbing", "ascending", "growing"],
    },
    "decreasing": {
        "verb":    ["fall", "decrease", "drop", "descend", "decline"],
        "noun":    ["fall", "decrease", "descent", "drop", "downward movement"],
        "gerund":  ["falling", "decreasing", "dropping", "descending", "declining"],
    },
}

MAGNITUDE = {
    "steep":   {
        "adverb": ["steeply", "sharply", "rapidly",
                   "significantly", "substantially"],
        "adj":    ["steep", "sharp", "rapid", "significant", "substantial"],
    },
    "shallow": {
        "adverb": ["gently", "gradually", "slowly", "modestly", "slightly"],
        "adj":    ["gentle", "gradual", "slow", "modest", "slight"],
    },
    "neutral": {
        "adverb": ["", "noticeably", "measurably"],
        "adj":    ["", "noticeable", "measurable"],
    },
}

IRRELEVANT_TEMPLATES = [
    "The signal {phrase}.",
    "This function {phrase}.",
    "The waveform {phrase}.",
    "Notably, this signal {phrase}.",
    "In terms of its global properties, the signal {phrase}.",
    "What can be observed is that the waveform {phrase}.",
    "A feature worth noting is that the signal {phrase}.",
    "It is the case that this function {phrase}.",
    "The measured signal {phrase}.",
    "{phrase} — this is a property of the signal.",
]

IRRELEVANT_PHRASES = {
    "periodicity": [
        "repeats after a fixed interval",
        "completes a full cycle periodically",
        "returns to its previous value after one period",
        "oscillates with a constant frequency",
        "exhibits periodic behaviour",
    ],
    "boundedness": [
        "remains bounded between its minimum and maximum values",
        "does not exceed its amplitude",
        "is confined within a fixed range",
        "stays within a fixed interval at all times",
        "has a finite amplitude",
    ],
    "smoothness": [
        "is continuously differentiable everywhere",
        "has no discontinuities",
        "varies smoothly at every point",
        "has a well-defined derivative at this location",
        "changes without any abrupt transitions",
    ],
    "domain": [
        "is defined for all real-valued inputs",
        "has a domain spanning all real numbers",
        "is well defined at every point along the axis",
        "takes real values across its entire domain",
        "can be evaluated at any point",
    ],
}

# ---------------------------------------------------------------------------
# Transition templates
# Used when the signal is about to drop to -0.5 (flatline) or recover from it.
# Both template sets are always future-tense.
# ---------------------------------------------------------------------------

DROP_WORD_BANKS = {
    "verb":   ["cut", "clamp", "collapse", "go", "transition"],
    "noun":   ["cut", "clamp", "collapse", "transition", "descent"],
    "gerund": ["cutting", "clamping", "collapsing", "going", "transitioning"],
}

RECOVERY_WORD_BANKS = {
    "verb":   ["recover", "resume", "return", "emerge", "revive"],
    "noun":   ["recovery", "resumption", "return", "emergence", "revival"],
    "gerund": ["recovering", "resuming", "returning", "emerging", "reviving"],
}

DROP_TEMPLATES = [
    # 0 — simple
    "The signal will no longer follow {signal} and instead soon {verb} to a negative constant.",
    # 1 — imminent
    "The signal is not a {signal} and is about to {verb} to -0.5.",
    # 2 — expectation
    "Expect the signal to change from a {signal} and {verb} to negative horizontal line shortly.",
    # 3 — nominal
    "A {noun} to a constant value is imminent due to a change of signal from a {signal}.",
    # 4 — gerund-led
    "{gerund} to minus 0.5, the signal will shortly become flat instead of following a {signal}.",
    # 5 — existential
    "A new signal, different to {signal}, will shortly be {noun} to negative half.",
    # 6 — signal's behaviour
    "The signal's behaviour will change from {signal} and shortly become a {noun} to minus half.",
    # 7 — temporal clause
    "In the near term, the signal will {verb} to - 0.5 rather than following a {signal}.",
    # 8 — wh-nominal
    "What will characterise the signal next is no longer a {signal} but a {noun} to a flatline.",
    # 9 — passive-adjacent
    "The signal is not a {signal} anymore and will shortly be observed to {verb} to a flat period.",
]

RECOVERY_TEMPLATES = [
    # 0
    "The signal will soon {verb} from -0.5 back to a {signal}.",
    # 1
    "The signal is about to {verb} its oscillation as a {signal}.",
    # 2
    "Expect the signal to {verb} from its flat period shortly.",
    # 3
    "A {noun} from negative half to follow a {signal} is imminent.",
    # 4
    "{gerund} from a negative horizontal line, the {signal} will shortly oscillate again.",
    # 5
    "There will shortly be a {noun} from minus 0.5 to a {signal}.",
    # 6
    "The signal's behaviour will shortly become a {signal}, {noun} from negative.",
    # 7
    "In the near term, the {signal} will {verb} from its flat period.",
    # 8
    "What will characterise the {signal} next is a {noun} from constant value.",
    # 9
    "The {signal} will shortly be observed to {verb} from negative 0.5.",
]


# ---------------------------------------------------------------------------
# Category specifications
# ---------------------------------------------------------------------------
# Each entry defines:
#   dir_correct  : bool — does the direction match the true signal direction?
#   mag_correct  : bool — does the magnitude match the true signal magnitude?
#   is_irrelevant: bool — ignore direction/magnitude entirely

CATEGORY_SPECS = {
    "correct":  {"dir_correct": True,  "mag_correct": True,  "is_irrelevant": False},
    "incorrect": {"dir_correct": False, "mag_correct": False, "is_irrelevant": False},
    "irrelevant":        {"dir_correct": None,  "mag_correct": None,  "is_irrelevant": True},
}

@dataclass
class TextSample:
    """A single generated text string with its full metadata."""
    text:           str
    template_idx:   int        # which TEMPLATES entry was used (0–9)
    tense:          str        # past | present | future
    direction_word: str        # the actual word chosen for direction slot
    magnitude_word: str        # the actual word chosen for magnitude slot
    irrelevant_group: Optional[str] = None  # periodicity | boundedness | etc.


@dataclass
class AnnotationPoint:
    """
    All text samples generated for a single (t, category, tense) combination.
    Multiple TextSamples per point allow per-point variance estimation.
    """
    t:              float
    y:              float
    dy:             float
    phase:          str        # ascending_zero | peak | descending_zero | trough
    true_direction: str        # increasing | decreasing
    true_magnitude: str        # steep | shallow
    category:       str        # key from CATEGORY_SPECS
    tense:          str        # past | present | future
    samples:        list[TextSample] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def _true_direction(y:float, dy: float) -> str:
    return "increasing" if dy > 0.1 or (0.1>dy>-0.1 and y<0) else "decreasing"


def _true_magnitude(dy: float, steep_threshold: float = 0.7) -> str:
    return "steep" if abs(dy) > steep_threshold else "shallow"


def _opposite_direction(direction: str) -> str:
    return "decreasing" if direction == "increasing" else "increasing"


def _opposite_magnitude(magnitude: str) -> str:
    return "shallow" if magnitude == "steep" else "steep"


def _build_irrelevant_text(
    template_idx: int,
    rng: random.Random,
) -> tuple[str, str]:
    """
    Returns (text, group_name) for an irrelevant annotation.
    Uses IRRELEVANT_TEMPLATES, ignoring direction/magnitude slots.
    """
    group = rng.choice(list(IRRELEVANT_PHRASES.keys()))
    phrase = rng.choice(IRRELEVANT_PHRASES[group])
    template = IRRELEVANT_TEMPLATES[template_idx % len(IRRELEVANT_TEMPLATES)]
    text = template.format(phrase=phrase)
    return text, group


def _build_directional_text(
    template_idx: int,
    tense: str,
    direction: str,    # increasing | decreasing
    magnitude: str,    # steep | shallow | neutral
    rng: random.Random,
    signal_words: list = SIGNAL_WORDS,
) -> tuple[str, str, str]:
    """
    Fills a directional template.
    Returns (text, direction_word_used, magnitude_word_used).
    """
    dir_bank = DIRECTION[direction]
    mag_bank = MAGNITUDE[magnitude]

    verb    = rng.choice(dir_bank["verb"])
    noun    = rng.choice(dir_bank["noun"])
    gerund  = rng.choice(dir_bank["gerund"])
    adverb  = rng.choice(mag_bank["adverb"])
    adj     = rng.choice(mag_bank["adj"])
    time_w = rng.choice(TIME_WORDS[tense])
    time_clause  = rng.choice(TIME_CLAUSES[tense])
    signal = rng.choice(signal_words)

    # Strip trailing spaces if magnitude is empty string
    def clean(s: str) -> str:
        return " ".join(s.split())
    
    text = clean(TEMPLATES[template_idx].format(
        verb=verb, noun =noun, adverb=adverb,
        gerund = gerund, adj= adj, 
        time_w=time_w, time_clause=time_clause, signal=signal
    ))

    return text, verb, adverb


def build_annotation_point(
    t:              float,
    y:              float,
    dy:             float,
    phase:          str,
    category:       str,
    tense:          str,
    n_samples:      int = 5,
    seed:           Optional[int] = 42,
    steep_threshold: float = 0.7,
) -> AnnotationPoint:
    """
    Build an AnnotationPoint with n_samples TextSamples for the given
    (category, tense) combination at a single time-series location.

    Parameters
    ----------
    t, y, dy        : time-series values at this point
    phase           : one of ascending_zero | peak | descending_zero | trough
    category        : key from CATEGORY_SPECS
    tense           : past | present | future
    n_samples       : how many distinct text realisations to generate
    seed            : for reproducibility
    steep_threshold : |dy| above this is "steep", below is "shallow"

    Returns
    -------
    AnnotationPoint with n_samples TextSamples in .samples
    """
    if category not in CATEGORY_SPECS:
        raise ValueError(f"Unknown category '{category}'. "
                         f"Valid: {list(CATEGORY_SPECS.keys())}")
    if tense not in TIME_WORDS:
        raise ValueError(f"Unknown tense '{tense}'. Valid: {list(TIME_WORDS.keys())}")

    rng = random.Random(seed)
    spec = CATEGORY_SPECS[category]

    true_dir = _true_direction(y,dy)
    true_mag = _true_magnitude(dy, steep_threshold)

    point = AnnotationPoint(
        t              = round(t,  4),
        y              = round(y,  4),
        dy             = round(dy, 4),
        phase          = phase,
        true_direction = true_dir,
        true_magnitude = true_mag,
        category       = category,
        tense          = tense,
    )

    # Use each template at least once before repeating
    template_indices = list(range(len(TEMPLATES)))
    rng.shuffle(template_indices)
    #template_indices = rng.sample(template_indices,n_samples)

    for i in range(n_samples):
        template_idx = template_indices[i % len(template_indices)]

        if spec["is_irrelevant"]:
            text, group = _build_irrelevant_text(template_idx, rng)
            point.samples.append(TextSample(
                text             = text,
                template_idx     = template_idx,
                tense            = tense,
                direction_word   = "",
                magnitude_word   = "",
                irrelevant_group = group,
            ))

        else:
            # Resolve which direction and magnitude to use
            if spec["dir_correct"]:
                direction = true_dir
            else:
                direction = _opposite_direction(true_dir)

            if spec["mag_correct"]:
                magnitude = true_mag
            else:
                magnitude = _opposite_magnitude(true_mag)

            text, dir_word, mag_word = _build_directional_text(
                template_idx, tense, direction, magnitude, rng,
            )
            point.samples.append(TextSample(
                text           = text,
                template_idx   = template_idx,
                tense          = tense,
                direction_word = dir_word,
                magnitude_word = mag_word,
            ))

    return point


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------


def build_transition_annotation_point(
    t:               float,
    y:               float,
    dy:              float,
    phase:           str,
    category:        str,
    n_samples:       int = 5,
    seed:            Optional[int] = None,
) -> AnnotationPoint:
    """
    Build a single AnnotationPoint for a transition event.

    transition_type : 'will_drop_to_constant' | 'will_recover_from_constant'

    Text is always future-tense: it describes what the signal will do next
    (drop to flatline, or recover from flatline), not its current behaviour.
    Each of the n_samples uses a different template and different word choices.
    """
    rng = random.Random(seed)
    if phase in ('drop_to_constant', 'recover_from_constant'):
        templates = (DROP_TEMPLATES     if phase == 'drop_to_constant'
                    else RECOVERY_TEMPLATES)
        banks     = (DROP_WORD_BANKS    if phase == 'drop_to_constant'
                    else RECOVERY_WORD_BANKS)
    else:

        templates = (DROP_TEMPLATES     if rng.uniform() >0.5
                    else RECOVERY_TEMPLATES)
        banks     = (DROP_WORD_BANKS    if rng.uniform() >0.5
                    else RECOVERY_WORD_BANKS)

    point = AnnotationPoint(
        t              = round(t,  4),
        y              = round(y,  4),
        dy             = round(dy, 4),
        phase          = phase,
        true_direction = _true_direction(y,dy),
        true_magnitude = _true_magnitude(dy),
        category       = category,
        tense          = 'future',
    )

    template_indices = list(range(len(templates)))
    rng.shuffle(template_indices)
    #template_indices = rng.sample(template_indices,n_samples)

    for i in range(n_samples):
        tidx   = template_indices[i % len(template_indices)]

        verb   = rng.choice(banks["verb"])
        noun   = rng.choice(banks["noun"])
        gerund = rng.choice(banks["gerund"])
        signal = rng.choice(SIGNAL_WORDS)
        text   = templates[tidx].format(verb=verb, noun=noun, gerund=gerund, signal=signal)

        point.samples.append(TextSample(
            text           = text,
            template_idx   = tidx,
            tense          = 'future',
            direction_word = verb,
            magnitude_word = '',
        ))

    return point


_TRANSITION_PHASES = ('drop_to_constant', 'recover_from_constant')

# Probability that an incorrect point uses a random transition
# template instead of the normal directional template. Set to match the
# fraction of actual transition points in correct (~30%).
# but only selected for phases that AREN'T transitions
INCORRECT_TRANSITION_PROB = 0.3


def build_data_point(t, y, dy, phase="ascending_zero", n_samples=5, seed=None):
    data = {}

    is_transition = phase in _TRANSITION_PHASES

    for idx, category in enumerate(CATEGORY_SPECS):
        tense        = 'future'
        derived_seed = (seed + idx) if seed is not None else None

        if is_transition and category == 'correct':
            point = build_transition_annotation_point(
                t=t, y=y, dy=dy,
                phase=phase,
                category=category,
                n_samples=n_samples,
                seed=derived_seed,
            )
        elif category == 'incorrect':
            rng = random.Random(derived_seed)
            if rng.random() < INCORRECT_TRANSITION_PROB:
                available_phases = [x for x in _TRANSITION_PHASES if x!=phase]
                fake_phase = rng.choice(available_phases)
                point = build_transition_annotation_point(
                    t=t, y=y, dy=dy,
                    phase=fake_phase,
                    category=category,
                    n_samples=n_samples,
                    seed=derived_seed,
                )
                point.phase = phase  # restore true signal phase for analysis
            else:
                point = build_annotation_point(
                    t=t, y=y, dy=dy,
                    phase=phase,
                    category=category,
                    tense=tense,
                    n_samples=n_samples,
                    seed=derived_seed,
                )
        else:
            point = build_annotation_point(
                t=t, y=y, dy=dy,
                phase=phase,
                category=category,
                tense=tense,
                n_samples=n_samples,
                seed=derived_seed,
            )
        data[(category, tense)] = point
    return data