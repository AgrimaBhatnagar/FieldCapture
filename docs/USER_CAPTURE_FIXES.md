# FieldCapture capture fixes

## Changes in this build

1. Nameplate OCR now uses the full preprocessing pass by default in the Python capture pipeline. This improves the chance of finding product identifiers on labels where some thresholding variants fail.
2. Identifier confidence combines agreement across OCR passes with token level OCR confidence, rather than penalising every failed preprocessing variant equally.
3. Missing serial numbers are recorded as `not_visible`. A missing serial number alone no longer causes the entire nameplate capture to be rejected.
4. The parser recognises item identifiers as product identifiers when the label uses `ITEM` rather than `MODEL`.
5. The legacy damage tree no longer creates a false specific damage category for low confidence results. It stores the photograph as accepted for human review, with `damage_type.value` left null and `review_required=true`.
6. Very dark or nearly uniform images still request a retake.

## Validation on the three user supplied photographs

The Feit Electric carton photograph was processed successfully:

- Manufacturer: FEIT ELECTRIC
- Product identifier: ADJ8000/5K/MM/LED
- Serial: not visible / not detected
- Capture: accepted, no retake required

The floor and wall photographs were accepted for human review. The legacy classifier's predicted class was not stored as a verified damage type. This avoids falsely recording a water stain when the visible conditions include unfinished flooring, peeling paint or plaster deterioration.

## Important limitation

These changes fix capture handling and unsafe fallback behaviour. They do not train a new damage model and do not demonstrate greater than 90% damage classification accuracy. The legacy model remains a human review fallback until a properly trained and property separated model is available.
