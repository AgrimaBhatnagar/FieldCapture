# FieldCapture submission audit

Audit date: 2026-11-24
Audited artifact: `FieldCapture_Capture_Fixes` source package, with the OCR loading hint removed from `web/app.html`.

## Automated validation performed

- Python compilation: `python -m compileall -q app.py fieldcapture eval tests` completed successfully.
- Unit tests: `python -m unittest discover -s tests -v` completed with **60 tests passing**.
- The test run emitted ResourceWarnings for unclosed resources in test and application paths, and an `InconsistentVersionWarning` when loading a scikit-learn DecisionTreeClassifier saved with version 1.9.1 under version 1.8.0. Tests passed, but these warnings should be cleaned up in a production hardening pass.
- Three new tests verify that the trained damage checkpoint is packaged, runs through the capture inference function, and abstains on a blank frame.
- The UI loading hint saying first-time OCR can take 5 to 20 seconds was removed from `web/app.html`.

## Previously measured reference evaluation

- Nameplate OCR: 24/30 exact fields (80%) on 10 reference nameplate images.
- Trained Random Forest damage presence model: 78.05% leave one out overall accuracy on 41 development images. At a 0.70 threshold, the historical selective score was 95.83% on 24 accepted predictions. This is not an independent held out result. The subtype tree remains advisory and requires human review.
- Image quality gate: 46/51 accepted (90.2%). This is a quality gate result, not classification accuracy.
- The reference images are not a held out field dataset. The damage labels do not include verified physical property IDs, so property separated generalisation cannot be calculated honestly. The trained model detects binary damage presence only, not crack, stain, or standing water subtypes.

## Not validated by this audit

- No real Meta glasses or Mock Device Kit session was run.
- The Swift integration scaffold was not compiled in Xcode. It contains comments marking API details that require verification.
- No ResNet18 checkpoint was trained or evaluated. The packaged trained model is a Random Forest for binary damage presence, not a deep learning subtype classifier.
- PaddleOCR was not confirmed running in this environment. The optional adapter is code, not evidence of OCR improvement.
- The browser camera, microphone and live speech path were not manually exercised on the user's laptop in this audit.
- No real property walkthroughs, three genuine job records or demo video were available.
- The research memo template contains no measured hardware findings.

## Submission recommendation

This package can be submitted as a software prototype with an honest limitations section if the assignment accepts a prototype and clearly disclosed gaps. It should not be represented as a completed end to end Meta glasses implementation, as a field validated insurance product, or as a damage classifier with greater than 90% accuracy. If the assignment explicitly requires compiled DAT integration, original field captures, a measured hardware memo, three genuine job records and a demo video, those requirements remain unmet.
