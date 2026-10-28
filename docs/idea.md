# Hackathon idea: StrainSight

**Research question:** Can a measurement-aware point-query policy estimate localized deformation more accurately than an equally budgeted fixed point grid, with both approaches constrained to the same specimen mask?

## Workflow
1. Input a video of a material specimen being stretched, plus a reference-frame mask.
2. Generate masked, spatially distributed points for the baseline.
3. Track those material points through the video with a pretrained point tracker.
4. Fit a local displacement field and compute strain on reliable neighborhoods.
5. Score regions for refinement using strain-gradient magnitude, tracking uncertainty and coverage.
6. Allocate additional point queries, recompute the strain map, and visualize the improvement.
7. Expose computed measurements to an assistant built using Meta open-weight models.

## Experiments
- **Primary baseline:** uniform grid inside the *same* foreground mask.
- **Adaptive method:** same total point budget, with initial coarse queries and follow-up queries selected by measurements.
- **Metrics:** mean and 95th-percentile displacement error, strain MAE, high-strain-region error, runtime, VRAM, and query count.
- **Datasets:** investigate 2D-DIC Challenge 2.0 for reference measurements; test transfer to an AA5086 tensile-test sequence where data permissions and availability permit.

## Risks
- Visible texture and image resolution can limit tracking reliability.
- Initial mask error can exclude the very regions of interest.
- Strong localized strain and occlusions can corrupt tracks.
- Not all tensile datasets supply dense pixelwise strain ground truth.
- Masking itself is not adaptive tracking: novelty must be measured against a masked baseline.

## Immediate milestone
Load two consecutive frames and a reference mask, draw 100 foreground query points, then visualize correspondences from a pretrained tracker. Separate baseline from policy before tuning.
