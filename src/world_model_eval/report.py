"""Render a WorldModelReport as text, JSON, or a self-contained HTML page."""
from __future__ import annotations

import html
import json


def to_json(rep) -> str:
    return json.dumps({"tol": rep.tol, "channels": rep.channels,
                       "summary": rep.summary(),
                       "per_scenario": rep.per_scenario}, indent=2)


def to_text(rep) -> str:
    s = rep.summary()
    out = ["world-model-eval report",
           "  tolerance          %.4g" % rep.tol,
           "  scenarios          %d" % s["n_scenarios"],
           "  trust horizon      median %.0f steps  (min %d, max %d)"
           % (s["trust_steps_median"], s["trust_steps_min"], s["trust_steps_max"]),
           "  consistency        %s" % ", ".join("%s x%d" % (k, v)
                                                 for k, v in s["consistency"].items()),
           "  plausibility       %.0f%% of scenarios pass all checks"
           % (100 * s["plausibility_pass_rate"]),
           "  calibration        %s"
           % ("%.3f mean gap (0 perfect)" % s["calibration_mean"]
              if s["calibration_mean"] is not None else "n/a (no uncertainty)"),
           "",
           "  %-16s %6s %6s %-12s %5s %s" % ("scenario", "trustH", "final", "growth",
                                             "plaus", "calib")]
    for r in rep.per_scenario:
        out.append("  %-16s %6d %6.3f %-12s %5s %s"
                   % (r["scenario"][:16], r["trust_horizon"]["steps"], r["final_error"],
                      r["consistency"]["kind"],
                      "ok" if r["plausibility_failed"] == 0 else "FAIL",
                      ("%.3f" % r["calibration"]) if r["calibration"] == r["calibration"]
                      else "-"))
    return "\n".join(out)


def to_html(rep) -> str:
    s = rep.summary()
    rows = []
    for r in rep.per_scenario:
        col = "#2b8a3e" if r["plausibility_failed"] == 0 else "#b02a37"
        rows.append(
            "<tr><td>%s</td><td>%d</td><td>%.3f</td><td>%s</td>"
            "<td style='color:%s'>%s</td><td>%s</td></tr>" % (
                html.escape(r["scenario"]), r["trust_horizon"]["steps"], r["final_error"],
                html.escape(r["consistency"]["kind"]), col,
                "pass" if r["plausibility_failed"] == 0 else "FAIL",
                ("%.3f" % r["calibration"]) if r["calibration"] == r["calibration"] else "-"))
    return _HTML.replace("{{ROWS}}", "\n".join(rows)).replace(
        "{{TOL}}", "%.4g" % rep.tol).replace(
        "{{N}}", str(s["n_scenarios"])).replace(
        "{{MED}}", "%.0f" % s["trust_steps_median"]).replace(
        "{{PLAUS}}", "%.0f" % (100 * s["plausibility_pass_rate"]))


_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>world-model-eval report</title><style>
body{font-family:'Segoe UI',Helvetica,Arial,sans-serif;color:#1f2933;background:#fff;
margin:0;padding:2rem 1rem;line-height:1.5}main{max-width:820px;margin:0 auto}
h1{margin:0 0 .2rem}.sub{color:#6b7580;margin:0 0 1.2rem}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border-bottom:1px solid #e1e6ea;padding:.5rem .6rem;text-align:left}
th{color:#0b7285}
@media(prefers-color-scheme:dark){body{background:#0d1420;color:#e6eaee}
th,td{border-color:#2b3440}}
</style></head><body><main>
<h1>world-model-eval report</h1>
<p class="sub">tolerance {{TOL}} · {{N}} scenarios · median trust horizon {{MED}} steps · {{PLAUS}}% plausibility pass</p>
<table><tr><th>scenario</th><th>trust H</th><th>final err</th><th>growth</th>
<th>plausibility</th><th>calib gap</th></tr>
{{ROWS}}
</table></main></body></html>"""
