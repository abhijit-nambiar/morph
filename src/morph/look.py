"""Natural-language look -> starting recipe (global + per-region sliders).

This is the local stand-in for a generative editor: it turns photographic vocabulary into
slider moves, understands region targeting ("soft skin", "darker background", "brighter eyes",
"richer hair") and intensity modifiers ("slightly", "very", "less", "no").  Unknown words are
reported back so the UI can show what was and wasn't understood.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .recipe import Recipe

REGION_WORDS = {
    "face_skin": ["face", "facial", "complexion", "faces"],
    "body_skin": ["body", "arms", "legs", "body skin"],
    "skin": ["skin", "skintone", "skin tone", "skin tones", "skintones"],
    "hair": ["hair"],
    "eyes": ["eyes", "eye", "iris", "irises"],
    "lips": ["lips", "lip", "mouth"],
    "clothes": ["clothes", "clothing", "outfit", "outfits", "dress", "shirt", "suit", "wardrobe", "garments"],
    "subject": ["subject", "subjects", "people", "person", "portrait", "model", "couple"],
    "background": ["background", "backdrop", "surroundings", "environment", "scene behind"],
    "sky": ["sky", "skies", "clouds"],
    "vegetation": ["greenery", "foliage", "grass", "trees", "leaves", "plants", "greens"],
}

# term -> {key: value}; keys "region.key" are local, "@" means "the region named in the clause".
TERMS: dict[str, dict[str, float]] = {
    "warm": {"temperature": 0.25, "@tint": 0.02},
    "warmer": {"temperature": 0.25},
    "cool": {"temperature": -0.25},
    "cooler": {"temperature": -0.25},
    "cold": {"temperature": -0.35, "saturation": -0.05},
    "bright": {"exposure": 0.35, "shadows": 0.1},
    "brighter": {"exposure": 0.35},
    "brighten": {"exposure": 0.35},
    "lighter": {"exposure": 0.3},
    "airy": {"exposure": 0.35, "shadows": 0.3, "contrast": -0.15, "blacks": 0.15, "saturation": -0.08},
    "dark": {"exposure": -0.35},
    "darker": {"exposure": -0.35},
    "darken": {"exposure": -0.35},
    "moody": {"exposure": -0.25, "contrast": 0.15, "highlights": -0.25, "saturation": -0.15, "grade_shadows_x": -0.15, "vignette": -0.25, "blacks": -0.1},
    "dramatic": {"contrast": 0.35, "clarity": 0.2, "highlights": -0.3, "shadows": -0.1, "vignette": -0.2},
    "contrast": {"contrast": 0.3},
    "contrasty": {"contrast": 0.35},
    "punchy": {"contrast": 0.3, "vibrance": 0.2, "clarity": 0.1},
    "crisp": {"clarity": 0.2, "texture": 0.15, "sharpen": 0.3, "dehaze": 0.05},
    "sharp": {"texture": 0.2, "sharpen": 0.4},
    "flat": {"contrast": -0.3, "highlights": -0.2, "shadows": 0.2},
    "matte": {"blacks": 0.3, "curve_shadows": 0.25, "contrast": -0.1},
    "faded": {"blacks": 0.3, "curve_shadows": 0.25, "saturation": -0.15, "contrast": -0.1},
    "soft": {"contrast": -0.15, "clarity": -0.15, "highlights": -0.1},
    "dreamy": {"clarity": -0.3, "dehaze": -0.15, "highlights": 0.1, "exposure": 0.1, "contrast": -0.15},
    "hazy": {"dehaze": -0.25},
    "clear": {"dehaze": 0.15, "clarity": 0.1},
    "vibrant": {"vibrance": 0.35, "saturation": 0.1},
    "colorful": {"vibrance": 0.3, "saturation": 0.1},
    "saturated": {"saturation": 0.3},
    "rich": {"vibrance": 0.2, "saturation": 0.1, "blacks": -0.1},
    "richer": {"saturation": 0.2, "blacks": -0.05},
    "muted": {"saturation": -0.25, "vibrance": -0.1},
    "desaturated": {"saturation": -0.4},
    "pastel": {"saturation": -0.2, "exposure": 0.15, "blacks": 0.2, "contrast": -0.15},
    "monochrome": {"saturation": -1.0},
    "bw": {"saturation": -1.0, "contrast": 0.15},
    "natural": {"vibrance": 0.05},
    "clean": {"dehaze": 0.08, "noise_reduction": 0.25, "whites": 0.08},
    "golden": {"temperature": 0.3, "tint": 0.04, "grade_highlights_x": 0.25, "grade_highlights_y": 0.25, "hsl_orange_sat": 0.1},
    "sunset": {"temperature": 0.35, "tint": 0.08, "grade_highlights_x": 0.3, "grade_highlights_y": 0.2},
    "cinematic": {"grade_shadows_x": -0.3, "grade_shadows_y": -0.15, "grade_highlights_x": 0.25, "grade_highlights_y": 0.2, "contrast": 0.15, "saturation": -0.1, "vignette": -0.15},
    "teal": {"grade_shadows_x": -0.35, "grade_shadows_y": -0.2},
    "orange": {"grade_highlights_x": 0.25, "grade_highlights_y": 0.25},
    "film": {"blacks": 0.2, "curve_shadows": 0.2, "curve_highlights": -0.15, "grain": 0.25, "saturation": -0.05, "grade_shadows_x": -0.1},
    "filmic": {"blacks": 0.2, "curve_shadows": 0.2, "curve_highlights": -0.15, "grain": 0.2},
    "vintage": {"blacks": 0.25, "temperature": 0.15, "saturation": -0.2, "grain": 0.25, "curve_highlights": -0.2},
    "grain": {"grain": 0.35},
    "grainy": {"grain": 0.45},
    "noise": {"noise_reduction": 0.5},  # "no noise" / "reduce noise" (negation handled below)
    "denoise": {"noise_reduction": 0.5},
    "smooth": {"@texture": -0.4, "@noise_reduction": 0.3, "@clarity": -0.1},
    "glowing": {"@exposure": 0.15, "@highlights": 0.1, "@clarity": -0.15, "@saturation": 0.05},
    "glow": {"@exposure": 0.15, "@highlights": 0.1, "@clarity": -0.15},
    "luminous": {"@exposure": 0.2, "@shadows": 0.15},
    "tan": {"@temperature": 0.12, "@saturation": 0.1, "@exposure": -0.1},
    "tanned": {"@temperature": 0.12, "@saturation": 0.1, "@exposure": -0.1},
    "rosy": {"@tint": 0.12, "@saturation": 0.08},
    "even": {"@contrast": -0.15, "@texture": -0.15, "@saturation": -0.05},
    "pop": {"@exposure": 0.15, "@contrast": 0.15, "@saturation": 0.15, "@clarity": 0.1},
    "shadows": {"shadows": 0.3},
    "lifted": {"shadows": 0.3, "blacks": 0.15},
    "lift": {"shadows": 0.3, "blacks": 0.15},
    "crushed": {"blacks": -0.35},
    "deep": {"blacks": -0.25, "shadows": -0.1},
    "highlights": {"highlights": -0.35},
    "recover": {"highlights": -0.4, "whites": -0.15},
    "exposure": {"exposure": 0.25},
    "vignette": {"vignette": -0.3},
    "magenta": {"tint": 0.15},
    "green": {"tint": -0.15},
    "greener": {"hsl_green_sat": 0.25, "hsl_yellow_hue": 0.1},
    "bluer": {"hsl_blue_sat": 0.25},
    "blue": {"hsl_blue_sat": 0.2},
    "emerald": {"hsl_green_hue": 0.2, "hsl_green_sat": 0.1, "hsl_green_lum": -0.15},
    "lush": {"hsl_green_sat": 0.25, "hsl_yellow_hue": 0.15, "hsl_green_lum": -0.1},
    "dehaze": {"dehaze": 0.3},
    "detail": {"texture": 0.2, "clarity": 0.1},
    "texture": {"texture": 0.25},
}
PHRASES = {
    "golden hour": "golden",
    "black and white": "bw",
    "b&w": "bw",
    "teal and orange": "cinematic",
    "teal orange": "cinematic",
    "high contrast": "contrasty",
    "low contrast": "flat",
    "less contrast": "flat",
    "lifted shadows": "lifted",
    "lift shadows": "lifted",
    "lift the shadows": "lifted",
    "open shadows": "lifted",
    "crushed blacks": "crushed",
    "deep blacks": "deep",
    "recover highlights": "recover",
    "pull highlights": "recover",
    "skin smoothing": "smooth",
    "noise reduction": "denoise",
    "reduce noise": "denoise",
    "no noise": "denoise",
    "less noise": "denoise",
    "clean skin": "smooth",
}
MODIFIERS = {
    "slightly": 0.5, "slight": 0.5, "a bit": 0.5, "a little": 0.5, "subtle": 0.5, "subtly": 0.5, "touch": 0.5, "mildly": 0.6,
    "somewhat": 0.7, "more": 1.0, "very": 1.6, "really": 1.5, "much": 1.5, "super": 1.8, "extremely": 2.0, "strong": 1.5,
    "strongly": 1.5, "heavy": 1.6, "heavily": 1.6, "no": 0.0, "without": 0.0, "less": -0.7, "reduce": -0.8, "reduced": -0.8, "lower": -0.8, "not": -1.0,
}
REGION_TERM_OVERRIDE = {("vegetation", "green"): "lush", ("vegetation", "greener"): "lush", ("sky", "blue"): "bluer", ("sky", "bluer"): "bluer"}
# region-specific defaults when a clause names only a region plus a generic word
REGION_DEFAULT_SCALE = {"eyes": 0.8, "lips": 0.7, "face_skin": 0.8, "body_skin": 0.8, "skin": 0.8, "hair": 0.9}
GLOBAL_ONLY = ("curve_", "hsl_", "grade_", "grain", "vignette", "sharpen", "vibrance")
STOP = {"the", "and", "a", "an", "with", "of", "to", "in", "on", "for", "look", "feel", "tones", "tone", "make", "keep",
        "but", "while", "it", "its", "all", "overall", "photos", "photo", "image", "images", "style", "light", "lighting",
        "colors", "colours", "color", "colour", "edit", "add", "give", "that", "is", "be", "bit", "little", "some", "up"}


@dataclass
class ParsedLook:
    prompt: str
    recipe: Recipe
    matched: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    t = text.lower().replace("-", " ").replace("’", "'")
    for p, rep in sorted(PHRASES.items(), key=lambda kv: -len(kv[0])):
        t = re.sub(rf"\b{re.escape(p)}\b", rep, t)
    for reg, words in REGION_WORDS.items():
        for w in sorted(words, key=len, reverse=True):
            if " " in w:
                t = t.replace(w, w.replace(" ", "_"))
    return t


def _regions_in(words: list[str]) -> list[str]:
    out = []
    for w in words:
        for reg, vocab in REGION_WORDS.items():
            if w.replace("_", " ") in vocab and reg not in out:
                out.append(reg)
    if "skin" in out:
        out.remove("skin")
        out += [r for r in ("face_skin", "body_skin") if r not in out]
    return out


def parse(prompt: str, strength: float = 1.0) -> ParsedLook:
    rec = Recipe(meta={"look": prompt})
    matched, unknown = [], []
    clauses = re.split(r"[,;.\n]|\band\b|\bbut\b|\bwith\b|\bwhile\b", _norm(prompt))
    for clause in clauses:
        words = re.findall(r"[a-z_&']+", clause)
        if not words:
            continue
        regions = _regions_in(words)
        mult, pending = 1.0, None
        for i, w in enumerate(words):
            two = " ".join(words[i : i + 2])
            if two in MODIFIERS:
                pending = MODIFIERS[two]
                continue
            if w in MODIFIERS:
                pending = MODIFIERS[w] if pending is None else pending * abs(MODIFIERS[w])
                continue
            w = REGION_TERM_OVERRIDE.get((regions[0] if regions else "", w), w)
            term = TERMS.get(w) or TERMS.get(w.rstrip("s")) or (TERMS.get(w[:-2]) if w.endswith("er") else None)
            if term is None:
                if w not in STOP and not _regions_in([w]):
                    unknown.append(w)
                continue
            m = (pending if pending is not None else mult) * strength
            if w in ("noise", "denoise"):
                m = abs(m)
            pending = None
            scale = min((REGION_DEFAULT_SCALE.get(r, 1.0) for r in regions), default=1.0)
            for k, v in term.items():
                local_key = k.lstrip("@")
                if regions and not k.startswith(GLOBAL_ONLY) and not local_key.startswith(GLOBAL_ONLY):
                    for r in regions:
                        d = rec.local.setdefault(r, {})
                        d[local_key] = d.get(local_key, 0.0) + v * m * scale
                elif not k.startswith("@"):
                    rec.global_[k] = rec.global_.get(k, 0.0) + v * m
                else:
                    for r in ("face_skin", "body_skin"):  # "@" terms with no region default to skin
                        d = rec.local.setdefault(r, {})
                        d[local_key] = d.get(local_key, 0.0) + v * m * 0.8
            matched.append(w if not regions else f"{w}[{'/'.join(regions)}]")
    return ParsedLook(prompt, rec.clamped().pruned(), matched, unknown)
