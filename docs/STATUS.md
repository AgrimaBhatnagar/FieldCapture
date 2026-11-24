# Submission status

This status reflects the files actually included in this package and the validation run recorded in `docs/SUBMISSION_AUDIT.md`.

| Requirement | Status | Evidence or remaining work |
|---|---|---|
| Local browser inspection application | Implemented; automated tests pass | `app.py`, `web/app.html`, `tests/` |
| Voice and typed command parsing | Implemented; unit tested | `fieldcapture/commands.py` |
| Image quality gate and retake prompts | Implemented; unit tested | `fieldcapture/quality.py` |
| Nameplate OCR and field verification | Implemented; reference tested | 24/30 exact fields on 10 reference nameplates; optional PaddleOCR runtime not validated |
| Damage recognition | Partial | Trained Random Forest binary presence model on 41 development images; 78.05% leave one out overall accuracy. Subtype suggestions remain legacy and require technician review |
| Property separated model evaluation | Framework implemented; not run | Actual property IDs and independently verified labels are missing |
| Offline queue and sync retry | Implemented; unit tested | `fieldcapture/outbox.py` |
| Job record schema and estimate mapping | Implemented; unit tested | `schema.json`, `fieldcapture/validate.py`, `fieldcapture/lineitems.py` |
| Meta Device Access Toolkit app | Not complete | iOS code is uncompiled; marked SDK calls need verification; no Android app |
| Mock Device Kit or real glasses test | Not performed | Requires Xcode, registered developer app and supported device/simulator |
| Original field evaluation set | Not complete | 40+ original frames and verified labels are not included |
| Three genuine inspection records | Not complete | Sample record is illustrative and not three real walk-throughs |
| Research memo with measured SDK findings | Not complete | Template and desk research only; no actual device measurements |
| Demo video | Not included | Must be recorded from a tested build |
| Browser hardware validation | Not established by unit tests | Run camera, microphone, speech, offline and recovery tests on the target laptop |

## Important model statement

A trained Random Forest damage presence checkpoint is included at `models/damage_presence_rf.pkl`. On 41 labelled development images, leave one out cross validation reported 78.05% overall accuracy. At a 0.70 confidence threshold, the historical report recorded 95.83% accuracy on 24 accepted predictions. This is selective accuracy, not overall accuracy, and the accepted subset was biased toward damage positive predictions. Labels were not independently verified and this is not a property separated or held out field test. The model detects damage presence only. The subtype suggestion is from a legacy tree and requires technician confirmation. Do not claim 90% or higher overall damage accuracy.

## Important hardware statement

The Swift files are an integration scaffold, not a working compiled application. Several API symbols, stream publisher locations and permission details are explicitly marked for verification. Do not claim Meta glasses integration is complete.
