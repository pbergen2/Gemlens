"""Dependency-free, deliberately conservative card screening baseline.

Runs unchanged in CPython/FastAPI and Pyodide. No trained model is implied.
Coordinates are normalized to the supplied image, not the physical card.
"""
import math
import statistics

VERSION = "0.2.0-heuristic"


def median(values):
    return statistics.median(values) if values else 0


def analyze(rgba, width, height, side="front", original_height=0, crop=None):
    w, h = int(width), int(height)
    if w < 80 or h < 80 or w * h > 400000 or len(rgba) != w * h * 4:
        raise ValueError("Image must contain 80–400,000 pixels per dimension product and valid RGBA data.")
    if side not in ("front", "back"):
        raise ValueError("Unknown card side.")

    def pixel(x, y):
        i = (int(y) * w + int(x)) * 4
        return rgba[i:i + 3]

    def dist(a, b):
        return sum(abs(a[i] - b[i]) for i in range(3)) / 3

    def lum(p):
        return sum(p) / 3

    warnings = []
    if crop is not None:
        if len(crop) != 4 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in crop):
            raise ValueError("Invalid boundary.")
        l, t, r, b = [float(v) for v in crop]
        if not (0 <= l < r <= 1 and 0 <= t < b <= 1 and r-l >= .2 and b-t >= .2):
            raise ValueError("Boundary is too small or outside the image.")
        x0, y0, x1, y1 = int(l*w), int(t*h), min(w-1, int(r*w)-1), min(h-1, int(b*h)-1)
        boundary = "manual"
    else:
        samples = [pixel(x, y) for x in (1, w-2) for y in (1, h-2)]
        bg = [median([p[c] for p in samples]) for c in range(3)]
        rows = []
        for y in range(2, h-2, 2):
            xs = [x for x in range(2, w-2, 2) if dist(pixel(x, y), bg) > 42]
            if len(xs) > w * .18:
                rows.append((y, xs[0], xs[-1]))
        if len(rows) < h*.15:
            return {"side": side, "valid": False, "reason": "No clear card boundary found. Use a dark, plain background with the full card visible.", "version": VERSION}
        middle = rows[len(rows)//4:3*len(rows)//4]
        x0, x1 = int(median([row[1] for row in middle])), int(median([row[2] for row in middle]))
        y0, y1 = rows[0][0], rows[-1][0]
        boundary = "automatic"
    cw, ch = x1-x0, y1-y0
    bbox = [x0/w, y0/h, (x1+1)/w, (y1+1)/h]
    if cw < 55 or ch < 75 or not .58 < cw/ch < .85:
        return {"side": side, "valid": False, "bbox": bbox, "reason": "The outline does not look like an upright card. Retake straight on, or adjust the boundary below.", "version": VERSION}
    if min(x0, y0, w-1-x1, h-1-y1) < 2:
        warnings.append("Card touches the image boundary; missing edges may hide damage.")
    if original_height and original_height * ch/h < 900:
        warnings.append("Card resolution is below 900 pixels tall. Small defects may be missed.")

    # Measure border transitions independently on opposing sides. Requiring
    # agreement across scanlines avoids turning a single artwork edge into a border.
    def border(axis, reverse):
        length = cw if axis == "x" else ch
        cross = ch if axis == "x" else cw
        positions, strengths = [], []
        for f in (.28, .36, .44, .52, .60, .68, .76):
            k = int(cross*f)
            def at(d):
                if axis == "x":
                    return pixel(x1-d if reverse else x0+d, y0+k)
                return pixel(x0+k, y1-d if reverse else y0+d)
            candidates = [(dist(at(d-2), at(d+2)), d) for d in range(max(3, int(length*.025)), max(4, int(length*.16)))]
            strength, pos = max(candidates)
            positions.append(pos)
            strengths.append(strength)
        position = median(positions)
        consistent = sum(abs(p-position) <= max(2, length*.013) for p in positions) >= 5
        return position, consistent and median(strengths) > 23

    borders = [border("x", False), border("y", False), border("x", True), border("y", True)]
    centering = None
    inner = None
    if all(v[1] for v in borders):
        left, top, right, bottom = [v[0] for v in borders]
        lr, tb = round(100*left/(left+right)), round(100*top/(top+bottom))
        imbalance = max(lr, 100-lr, tb, 100-tb)
        score = max(1, min(10, 10 - max(0, imbalance-52)*.18))
        centering = {"score": round(score, 1), "left_right": [lr, 100-lr], "top_bottom": [tb, 100-tb]}
        inner = [(x0+left)/w, (y0+top)/h, (x1-right)/w, (y1-bottom)/h]
    else:
        warnings.append("Inner border is ambiguous; centering was not scored. Full-art and irregular borders need manual measurement.")

    border_pixels, edge_pixels, corner_pixels = [], [], []
    for y in range(y0+2, y1-1):
        for x in range(x0+2, x1-1):
            nx, ny = (x-x0)/cw, (y-y0)/ch
            near = min(nx, 1-nx, ny, 1-ny)
            if .008 <= near <= .035:
                p = pixel(x, y)
                corner = min(nx, 1-nx) < .12 and min(ny, 1-ny) < .12
                border_pixels.append(p)
                (corner_pixels if corner else edge_pixels).append((x, y, p))
    chromatic = median([max(p)-min(p) for p in border_pixels]) > 35

    def whitening(points):
        flagged = [(x, y) for x, y, p in points if min(p) > 170 and max(p)-min(p) < 40]
        ratio = len(flagged)/max(1, len(points))
        return ratio, flagged

    edge_ratio, edge_flags = whitening(edge_pixels)
    corner_ratio, corner_flags = whitening(corner_pixels)
    if not chromatic:
        warnings.append("Neutral or silver border detected. Whitening cannot be separated reliably from printed color; edges and corners were not scored.")
    edge_score = round(max(1, 10-80*edge_ratio), 1) if chromatic else None
    corner_score = round(max(1, 10-65*corner_ratio), 1) if chromatic else None
    regions, total_regions = locate_whitening(
        edge_flags + corner_flags if chromatic else [], w, h, (x0, y0, x1, y1))
    if total_regions > len(regions):
        warnings.append(f"Showing the {len(regions)} largest of {total_regions} suspected patches.")
    # Global capture quality is kept separate from card damage and the grade.
    gray, lap = [], []
    glare = 0
    for y in range(y0+4, y1-4, 3):
        for x in range(x0+4, x1-4, 3):
            p = pixel(x, y)
            value = lum(p)
            gray.append(value)
            glare += int(min(p) > 247)
            lap.append(lum(pixel(x-1, y)) + lum(pixel(x+1, y)) + lum(pixel(x, y-1)) + lum(pixel(x, y+1)) - 4*value)
    sharpness = statistics.pvariance(lap) if len(lap) > 1 else 0
    poor = False
    if sharpness < 22:
        warnings.append("Image may be blurred or low-detail. Retake before relying on the estimate.")
        poor = True
    if median(gray) < 45:
        warnings.append("Image is underexposed. Retake with brighter, even lighting.")
        poor = True
    if glare/max(1, len(gray)) > .12:
        warnings.append("Large bright areas may be glare or printed artwork. Surface inspection requires another angle.")
    if regions:
        warnings.append("Highlighted spots are possible whitening, not confirmed damage. Check them against the original photo.")
    return {"version": VERSION, "side": side, "valid": True, "bbox": bbox, "inner_bbox": inner, "boundary": boundary,
            "centering": centering, "edges": edge_score, "corners": corner_score, "surface": None,
            "whitening_fraction": {"edges": round(edge_ratio, 4) if chromatic else None, "corners": round(corner_ratio, 4) if chromatic else None},
            "regions": regions, "warnings": warnings, "poor_capture": poor, "sharpness_proxy": round(sharpness, 1)}


def locate_whitening(flagged, w, h, card):
    """Bound connected candidate pixels; labels describe extent, not certainty."""
    x0, y0, x1, y1 = card
    cw, ch = x1-x0, y1-y0
    remaining = set(flagged)
    components = []
    for seed in sorted(remaining, key=lambda p: (p[1], p[0])):
        if seed not in remaining:
            continue
        remaining.remove(seed)
        points, stack = [seed], [seed]
        while stack:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = (x+dx, y+dy)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        points.append(neighbor)
                        stack.append(neighbor)
        if len(points) >= 3:
            components.append(points)
    components.sort(key=lambda ps: (-len(ps), min(p[1] for p in ps), min(p[0] for p in ps)))
    regions = []
    for points in components[:12]:
        left, right = min(p[0] for p in points), max(p[0] for p in points)+1
        top, bottom = min(p[1] for p in points), max(p[1] for p in points)+1
        x, y = (left+right)/2, (top+bottom)/2
        nx, ny = (x-x0)/cw, (y-y0)/ch
        extent = max((right-left)/cw, (bottom-top)/ch)
        severity = "light" if extent < .025 else "moderate" if extent < .08 else "pronounced"
        if min(nx, 1-nx) < .12 and min(ny, 1-ny) < .12:
            location = ("top" if ny < .5 else "bottom") + " " + ("left" if nx < .5 else "right") + " corner"
        else:
            location = min(((nx, "left edge"), (1-nx, "right edge"), (ny, "top edge"), (1-ny, "bottom edge")))[1]
        regions.append({"x": x/w, "y": y/h, "bbox": [left/w, top/h, right/w, bottom/h],
                        "type": "possible whitening", "severity": severity,
                        "label": f"Possible {severity} whitening", "location": location,
                        "extent_fraction": round(extent, 4), "candidate_pixels": len(points)})
    return regions, len(components)


def summarize(reports):
    valid = [r for r in reports if r.get("valid")]
    scores = []
    for r in valid:
        if r.get("centering"):
            scores.append(r["centering"]["score"])
        scores += [r[k] for k in ("edges", "corners") if r.get(k) is not None]
    complete = len(valid) == 2 and all(r.get("centering") and r.get("edges") is not None for r in valid)
    grade = None
    if scores and not any(r.get("poor_capture") for r in valid) and len(valid) == len(reports):
        anchor = min(scores)
        uncertainty = 1.5 if complete else 2.5
        grade = [max(1, math.floor(anchor-uncertainty)), min(10, math.ceil(anchor+1))]
    return {"grade_range": grade, "confidence": "Low — uncalibrated heuristic", "coverage": "front and back" if len(valid) == 2 else "one side only", "method": VERSION,
            "disclaimer": "Experimental screening, not a professional grade. Surface is unscored. Thresholds are not calibrated to any grading company.", "reports": reports}
