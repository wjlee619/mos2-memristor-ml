
---
## What I learned — Notebook 04

The electrode width result surprised me because I expected MoS2 layer count
to be the dominant variable. The data shows geometry dominates completely —
a process parameter I could control in fabrication, not just a material property.

Key intuition: think of the electrode as a door into the device. A 2 um door
has almost no area for current even if the MoS2 switches perfectly. An 18 um
door lets much more current through.

The 2 um devices did not fail randomly — they systematically showed only pA
current (noise floor). This means they never switched at all. In a real fab this
would be caught at L2 electrical test and binned as failures at L3 EDS.

The spatial map showing no positional clustering tells me MoS2 printing was
relatively uniform across the chip. Electrode geometry is the dominant variable,
not where on the chip the device sits.

---
## What I learned — Notebook 05

The 6 um result surprised me most. I expected it to be just "lower than
12 um" but the distribution shows it is actually two separate populations
— some devices that switched properly and some that never did. The mean
hides this completely.

This is exactly what my mentor meant by Anscombe's Quartet applied to fab
data. The number looks fine. The plot shows a problem.

In a real fab, if I reported "6 um electrode devices have mean i_on of
2.22 mA" to a process engineer, they would make the wrong decision. If I
showed them the bimodal distribution, they would immediately ask "what is
different about the devices in the lower population?" That question leads
to a root cause. The mean never gets you there.

---
## What I learned — Notebook 06

I went in thinking spray coating non-uniformity would be the answer.
Column 1 vs column 4 looked spatial, and I had a mental model of the
zigzag pattern leaving thin edges. But when I looked at the T-code
breakdown first, T24 had 90% failure vs T12 at 12%. That looked like
channel length was the culprit.

Then I realised: every T24 measurement happens to be at column 1.
Every column 4 measurement uses T12. The experiment was never designed
to separate these two variables. I can’t root-cause it.

This is what a confounded experiment looks like in real data. Both
hypotheses fit the numbers perfectly. The only way out is to run the
deconfounding experiment deliberately: measure T24 at column 4, and
T12 at column 1 with enough repeats.

The most useful thing I found: one T12 device at column 1 (FC1-T12,
Run 45) also failed. That’s a single data point, but it hints that
position might matter independently of T-code. If I had to bet, I’d
run FC1-T12 first — if it still fails, the spray coating hypothesis
survives. If it switches, T-code is the dominant factor.
