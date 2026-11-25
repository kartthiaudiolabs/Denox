# How scores are scaled

A composite, a system's score on the board that ranks it, is identified by its board, set ID, subset, anchor definition and resolved-calibration digests, and axis-membership version.
Compare composites only within that identity.
Adding or removing competitors cannot change another system's composite.
The 0.03 tie band of the composite and the tie band an axis declares (timbre: 0.05) remain display conventions; they are not confidence intervals.

## Fixed anchors

`anchors.json` declares raw-unit floors, ceilings and rationales.
Each sub-metric is `clip((value - floor) / (ceiling - floor), 0, 1)`.
Lower-is-better metrics have decreasing endpoints.
Average sub-metrics within an axis, axes within a part, and the parts equally.
`axes.json` defines three things:

- the four parts: what a system removes (suppression, music, cleanup, events, voices), keeps (preservation, fidelity), how the result sounds (timbre, quality) and what it restores (room);
- the boards, published side by side, each the mean of the parts it names and together naming every part exactly once (denoising = removes + keeps + sounds; restoration = restores); the first board orders the table;
- the axes scored and published beside the boards but included in neither (band).

A registry without parts averages its axes equally; one without boards publishes that composite alone.
The `*` after a system's name (the streaming probe) is not an axis and takes no weight: it is a yes or no read from a probe (`scorestream.py`: the future input the output depends on, and the drift over a long input), published beside the score as a screening, not a measure of delivery or throughput.

| Measurement | Floor → ceiling | Verified against |
|---|---|---|
| Babble suppression | 0 → −20 dB | INPUT reads 0 |
| Noisy SI-SDR | the nominal SNR (−5 or −6 dB) → +20 dB | INPUT reads the SNR |
| Primary speech deletion | 5% → 0% | |
| Quiet coherent voice retention | −10 → 0 dB | |
| Speaker similarity, at the moderate level of each of twelve noise families | 0.95 → 1.0 | INPUT (REF under crowd chatter) |
| Vocal sounds outside speech, lost | 30% → 0% | INPUT reads 0 |
| Clean speech: quiet coherent voice retention | −10 → 0 dB | INPUT reads 0 |
| Clean speech: level change | 6 → 1 dB | |
| Clean speech: high-band error | 3 → 0.5 dB | |
| Clean speech: speaker similarity | 0.95 → 1.0 | INPUT reads 1 |
| Clean speech: predicted quality | 4.0 → 4.5 MOS | INPUT |
| Clean speech: magnitude signal-to-error ratio | 10 → 30 dB | |
| Real room: speaker similarity to the dry voice | 0.86 → 1.0 | INPUT |
| Real room: predicted quality | 2.0 → 4.5 MOS | INPUT |
| Paired quality | 2.0 → 4.5 MOS | INPUT, REF |
| Reference-free quality | 2.4 → 5 MOS | INPUT |

Every endpoint is a declared constant, so one raw value maps to one score on every build and view of the set.
Where an endpoint stands for something the set can measure (the unprocessed input, the clean reference), freezing a calibration checks the measurement against the declared value within a stated tolerance and fails otherwise: a build that drifts needs a new anchor version, never a silently moved scale.
Scores from different builds still differ because their audio differs; the scale does not.

These constants express scoring policy, not demonstrated thresholds of audibility.
INPUT is zero only on INPUT-anchored rows; REF is one only on REF-anchored rows.
Neither has a meaningful ideal composite across all tasks.
Predicted quality can exceed the recording's quality and saturates at one.
Raw metrics remain available alongside the normalized axes.

`REF` renders clean references only.
Real recordings have no fabricated reference.
Freeze one `anchors.resolved.json` per set/subset; INPUT and REF supply the verification measurements.
Its deletion cohort is the INPUT-defined primary segment inventory.
Every ranked system must cover that complete cohort with the same segment denominators; missing data invalidates that system, without shrinking anyone else's cohort.
A missing/nonfinite measured anchor is an error.
Declared fallbacks apply only to measured but degenerate gaps.

A score file holds raw measurements; the anchor policy only scales them.
A change of policy re-stamps the score files and resolves a new calibration; it never re-scores.

The declared constants are policy, and each carries the range it could plausibly take.
How much of the order depends on these constants is measured in [METRICS.md](METRICS.md#how-much-of-the-order-is-policy).

Scores stamped with another anchor version are refused.
A changed constant needs a new anchor version and changelog.
