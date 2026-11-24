# FieldCapture

FieldCapture is a hands free field inspection prototype. It provides a local browser interface for camera or image upload, voice or typed inspection commands, nameplate OCR, damage capture, image quality checks, offline queuing, editable inspection fields, job JSON validation, and estimate line item mapping.

**Submission status: software prototype validated by automated tests, with a trained binary damage presence model. Not a completed Meta glasses product or a field validated damage recognition system.** Read `docs/SUBMISSION_AUDIT.md` and `docs/STATUS.md` before describing the project.

## Run locally on Windows

1. Install Python 3.11 or newer and the Tesseract OCR application.
2. Open this folder in VS Code and open a terminal.
3. Create and activate an environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

4. Open `http://localhost:8000` in Chrome or Edge. Allow camera and microphone permissions. You can also upload an image.

Example commands:

- `capture nameplate`
- `capture damage kitchen north wall`
- `note: subfloor soft`

On Windows, if `tesseract --version` is not recognised, add the Tesseract installation directory (commonly `C:\Program Files\Tesseract-OCR`) to PATH and reopen the terminal.

## Run tests

```powershell
python -m compileall -q app.py fieldcapture eval tests
python -m unittest discover -s tests -v
```

The current packaged source was tested with 57 passing unit tests. See `docs/SUBMISSION_AUDIT.md` for the exact scope and limitations. Passing unit tests do not establish real device compatibility or model accuracy.

## Architecture

- `app.py`: local HTTP application and browser interface.
- `web/app.html`: browser camera, image upload, command entry, capture results and inspection UI.
- `fieldcapture/commands.py`: voice and typed command parsing.
- `fieldcapture/quality.py`: image quality checks and retake prompts.
- `fieldcapture/ocr.py` and `fieldcapture/ocr_backend.py`: nameplate OCR, field parsing and optional PaddleOCR adapter.
- `fieldcapture/damage.py` and `fieldcapture/extract.py`: damage features and trained Random Forest presence inference, with a review only subtype suggestion.
- `models/damage_presence_rf.pkl`: trained binary damage versus no damage checkpoint.
- `eval/train_damage_presence.py`: reproducible training script for the bundled development references.
- `docs/DAMAGE_MODEL.md`: model behaviour, reported development results, and limitations.
- `fieldcapture/outbox.py`: durable offline queue, retry and deduplication.
- `fieldcapture/process.py`, `fieldcapture/validate.py`, `fieldcapture/lineitems.py`: capture processing, record validation and estimate code mapping.
- `eval/`: reference image results, evaluation scripts and training manifest template.
- `ios/FieldCapture/`: exploratory iOS Device Access Toolkit integration code. It is not compiled or verified against the SDK.

## Vision evaluation

The available reference set contains 51 images. The reported nameplate result is 24 of 30 exact fields (80%) on 10 reference nameplates. The trained Random Forest detects damage presence versus none using 41 labelled development images. Its leave one out overall accuracy was 78.05%; selective accuracy was 95.83% on 24 accepted predictions at a 0.70 confidence threshold. This selective score is not overall accuracy or independent field performance. Crack, stain, and standing water subtype suggestions remain unvalidated and require technician review. See `docs/DAMAGE_MODEL.md`.

## Known limitations

- No compiled or hardware tested Meta Device Access Toolkit application is included.
- The iOS integration is a scaffold with SDK symbols and stream details marked for verification.
- The trained model is binary damage presence detection only. It does not establish greater than 90% overall damage accuracy or independently classify damage subtypes.
- PaddleOCR is optional and has not been validated in the packaged runtime; Tesseract is the available fallback.
- The 51 image set is reference material, not 40 original Meta glasses inspection frames. Damage property IDs are not verified.
- The research memo is a template and secondary research summary, not a report of measured glasses testing.
- Three genuine field job records and a demo video are not included.
- Camera, microphone, browser permissions, real speech, network loss and offline recovery should be manually tested on the target laptop and browser.

Do not describe these uncompleted items as completed. The current deliverable is a tested software prototype and implementation scaffold.

### OCR and damage model notes

See `docs/OCR_MODEL.md` for the PaddleOCR and Tesseract ensemble, and `docs/DAMAGE_MODEL.md` for the conservative trained damage presence model and its evaluation limits. Install optional vision dependencies with `pip install -r requirements-vision.txt`.

## Additional AI model work

See `docs/MODEL_UPGRADE_AND_VALIDATION.md` for the four class property grouped
training workflow, stricter nameplate identifier verification, and PaddleOCR
versus Tesseract comparison command. The current reference data is not enough
to claim independent property validation: physical property IDs are missing.
PaddleOCR runtime comparison is also pending installation in a supported
Python 3.10 environment. See `docs/MODEL_UPGRADE_TEST_RESULTS.md` for the
actual environment test results.


## Reference image library

The reference selector is backed by `reference_images/annotations.json`. It currently contains 40 user annotated reference records: 10 appliance nameplates, 20 earlier property and wall damage images, and 10 additional property inspection images (`pic1_3.jpg` through `pic10_3.jpg`). Selecting a reference image uses its stored annotation record, including supplied dimensions and bounding boxes. Newly uploaded or captured images continue through the model inference pipeline.

Reference annotation confidence is a configured value for supplied reference data. It is not a measured model accuracy score. Physical dimensions are visual estimates unless independently measured on site.
