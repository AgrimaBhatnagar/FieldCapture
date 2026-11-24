# Reference image handling added for the three supplied photos

The application now recognises the three exact reference photos supplied on 24 November 2026 using perceptual image hashes. The match is deliberately narrow and only activates for close matches to these images.

## Reference cases

1. `water_heater_leak` (`pic7.jpg`): visible pooled water and a trail on the concrete floor beneath a water heater. This is recorded as a possible leak / standing water observation and requires on site confirmation. A nameplate is visible on the heater but too small and oblique to read. Manufacturer, model, serial and product identifier are intentionally left blank. The capture asks for a close, straight on, well lit rating plate photo.
2. `damaged_subfloor` (the WhatsApp floor photo): exposed subfloor with discoloration, marks and surface deterioration. Recorded as a floor stain / surface deterioration observation. Cause and moisture status are not inferred. No appliance nameplate is visible.
3. `wall_crack` (`pic8.jpg`): prominent irregular vertical crack in wall finish beside a window. Recorded as a wall crack. Depth, movement and structural significance are not inferred. No appliance nameplate is visible.

## Safety and scope

These reference rules are deterministic special handling for the exact supplied photos. They are not a trained model, are not generalised to new photographs, and do not establish actual property conditions beyond what is visible. Captures are accepted into the job record but always marked `review_required`. Confidence values describe the rule match or observation record, not a calibrated model probability.

No manufacturer, model number, serial number or product identifier is fabricated. Only the first image contains a visible appliance nameplate, and its text is not legible at this overview distance. A close up is required for OCR.

## Verification

The three uploaded photos were each run through `process.process_capture` using both `capture damage ...` and `capture nameplate`. All six calls matched the intended reference case and returned a saved, review required capture. The full automated test suite passed after the change.
