# Figure catalog

## Figure 1: `figure-01-quality-debiasing-frontier.pdf`

**Purpose.** Show that the target metric and response quality are in tension, that the
highest target-metric scores are produced by degenerate output, and that the clean
frontier is occupied by the CAD operators.

**Data source.** `sweep_points.csv`, all 87 points. Judged quality from
`judge_scores.json`; medicalization from the baseline runners (per response) and from
`rtsd_fullres_results-3.json` (aggregate, CAD only).

**Plotted variables.** x = mean medicalization score, axis inverted so more bias removed
is to the right. y = mean judged quality. Colour and marker = method. Filled = intact
output, hollow = degenerate. Error bars = percentile bootstrap 95% CI on the mean judged
quality, drawn only for intact points. Blue line = Pareto front over intact points at or
above neutral. Grey band = unsteered baseline +/- 2 SD across 10 no-steer points. Shaded
right region = overshoot past neutral.

**Caption requirements.** Must state n = 250 per point; that degeneracy is flagged from
text properties (repetition, unparseable output, overshoot) and not from the judge; that
the comparison is unpaired across sweep points; and that shading past neutral marks
withheld information rather than better debiasing.

**Key observation.** The intact points form a shallow curve that stays near the baseline
until roughly med = 0.2, then falls away. Every point that pushes to or past neutral is
hollow. SADI's published point sits at quality 3.19 while scoring well on the x-axis.

**Interpretation checklist.**
- Does the reader see that hollow points cluster at the high-debiasing end? 
- Is the neutral line read as a target rather than a midpoint?
- Are the four labelled published points traceable to their markers?
- Is it clear that CAA's published point removes slightly more bias than CAD's?

**Caveats.** Unpaired. No CI on the x-axis, and none at all for CAD. Overshoot shading
encodes a judgement (that passing neutral is undesirable) which the caption must state
explicitly rather than assume.

## Figure 2: `figure-02-collapse-and-cost.pdf`

**Purpose.** Panel (a) shows that collapse is a threshold effect present in every method
and that the thresholds differ. Panel (b) gives the quality cost at each method's own
published setting with uncertainty.

**Data source.** As above. Panel (a) uses one canonical configuration family per method
so that only strength varies: CAA L14, FairSteer thresh 0.5, SADI strengths, and the
three CAD operators.

**Plotted variables.** (a) x = steering strength in each method's own units, log scale;
y = mean judged quality; dotted line = unsteered baseline. (b) horizontal bars = change
in judged quality against baseline; error bars = bootstrap 95% CI.

**Caption requirements.** Must state that Angular is excluded from panel (a) because its
parameter is an angle and therefore circular, so it has no monotone dose ordering. Must
state that strengths are in each method's own units and are not comparable in magnitude
across methods, only in shape. Panel (b) must name the Holm correction.

**Key observation.** (a) Quality holds near baseline, then falls steeply. CAA and
FairSteer break between 2 and 4; the CAD operators hold to roughly 10 to 15; SADI breaks
between 5 and 10. (b) The cost at published settings spans -0.33 to -4.73, an order of
magnitude.

**Interpretation checklist.**
- Is the collapse read as a threshold rather than a gradual trade?
- Is it clear the x-axis compares shape and not absolute strength?
- Does panel (b) make the SADI outlier legible without implying the others are free?

**Caveats.** Panel (a) shows one configuration family per method, so it does not display
FairSteer's other thresholds or Angular's other strategies. Strength units differ by
method, and the log axis compresses that difference visually.

## Supporting table

`sweep_points.csv`: all 87 points with method, config, n, medicalization, judged quality
with 95% CI, change vs baseline, strict-recoded quality, unique ratio, parse-failure
rate, degeneracy flags.
