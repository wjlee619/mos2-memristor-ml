
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
