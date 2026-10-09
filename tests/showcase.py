#!/usr/bin/env python3
"""The showcase's tables against the documents: the six scenarios and their times must be the
README's performance table, the showcase README's and the chart's; the tour must use what exists."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_perf_charts  # noqa: E402

source = (ROOT / "examples/helixsr_showcase/main.c").read_text(encoding="utf-8")


def table(name, pattern):
    body = re.search(name + r"\[\] = \{(.*?)\};", source, re.S).group(1)
    return re.findall(pattern, body)


outputs = [(int(w), int(h)) for w, h in table("OUTPUTS", r"\{(\d+), (\d+), ")]
qualities = [(name, float(scale)) for name, scale in table("QUALITIES", r'\{"([^"]+)", "[^"]+", ([\d.]+)f\}')]
output_ids = re.search(r"enum \{ (OUT_\w+(?:, OUT_\w+)*), OUTPUT_COUNT \}", source).group(1).split(", ")
quality_ids = re.search(r"enum \{ (Q_\w+(?:, Q_\w+)*), QUALITY_COUNT \}", source).group(1).split(", ")
assert len(outputs) == len(output_ids) == 3 and len(qualities) == len(quality_ids) == 5, (outputs, qualities)
assert max(scale for _, scale in qualities) <= 3.0, "the runtime runs ratios up to 3x"

scenarios = []
for output, quality, ms in table("SCENARIOS", r"\{(OUT_\w+), (Q_\w+), ([\d.]+)\}"):
    (ow, oh), (mode, scale) = outputs[output_ids.index(output)], qualities[quality_ids.index(quality)]
    # the app truncates in single precision, as FidelityFX does
    render = (int(ow / scale + 1e-4), int(oh / scale + 1e-4))
    scenarios.append((f"{render[0]}×{render[1]}", f"{ow}×{oh}", mode, float(ms)))
assert len(scenarios) == 6, scenarios


def rows(path):
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\| (\d+×\d+) → (\d+×\d+) \| ([A-Za-z ]+?) \| ([\d.]+)(?: ms)? \|", line)
        if match:
            found.append((match.group(1), match.group(2), match.group(3), float(match.group(4))))
    return found


assert rows(ROOT / "README.md") == scenarios, (rows(ROOT / "README.md"), scenarios)
assert rows(ROOT / "examples/helixsr_showcase/README.md") == scenarios, rows(ROOT / "examples/helixsr_showcase/README.md")
assert [(render, output, mode, ms) for output, render, mode, ms, _ in build_perf_charts.CASES] == scenarios

shots = len(re.findall(r"^    \{\{", re.search(r"SHOTS\[\] = \{(.*?)^\};", source, re.S | re.M).group(1), re.M))
chapters = table("TOUR", r"(OUT_\w+), (Q_\w+), (V_\w+), (\d), (\d+), (\d+)\}")
assert len(chapters) == 10, len(chapters)
for output, quality, view, lens, shot, seconds in chapters:
    assert output in output_ids and quality in quality_ids and int(shot) < shots and int(seconds) >= 8, (output, quality, shot)
    assert not (output == "OUT_4K" and quality == "Q_NATIVE"), "4K has no native mode in the app"
assert "fsr4" not in source.lower().replace("ps5 fsr4 showcase", "").replace("ps5-fsr4", ""), "a leftover of the FSR4 app"
for chart in ("cases-light.svg", "cases-dark.svg"):
    theme = build_perf_charts.THEMES[chart[6:-4]]
    assert (ROOT / "docs/perf" / chart).read_text(encoding="utf-8") == build_perf_charts.cases_chart(theme), \
        f"docs/perf/{chart} is stale: run tools/build_perf_charts.py"
assert len(build_perf_charts.accuracy_rows()) == 24, "VALIDATION.md's results table changed shape"
for chart in ("accuracy-light.svg", "accuracy-dark.svg"):
    theme = build_perf_charts.THEMES[chart[9:-4]]
    assert (ROOT / "docs/perf" / chart).read_text(encoding="utf-8") == build_perf_charts.accuracy_chart(theme), \
        f"docs/perf/{chart} is stale: run tools/build_perf_charts.py"
for image in ("ps5-helixsr-showcase", "showcase-reading", "showcase-tower", "showcase-menu", "showcase-benchmark",
              "showcase-4k"):
    assert (ROOT / "docs/images" / f"{image}.png").is_file(), image
print("showcase tables: 6 scenarios, 10 chapters, documents and chart agree")
