# Product judgment: what I built for, and what I cut
Design target: a tech in gloves, in an attic, with no signal, who will not tolerate a fussy app.
- Big tap targets (48px+) and spoken feedback, so the phone can stay in a pocket. Typed box is a fallback, not the main path.
- Retake prompts say the fix ("too dark, turn on a light"), not the metric. They only fire on photo captures, never on notes.
- Capture-first, extract-later: the outbox writes before any upload, so no signal never blocks work.
- A failed frame is rejected, not saved; the tech gets one spoken sentence and tries again.
- Missing size gives qty null and low confidence instead of a guessed number, so estimators see what is unknown.
Cut, and why:
- Auto-OCR in the browser prototype: would need a model call I cannot verify offline; fields stay empty at confidence 0.
- Prices on line items: estimator's job, and invented prices would mislead.
- Sign-in, multi-job management, photo annotation: not needed to prove the capture loop.
- Wake-word handling: "capture ..." is the trigger; false-trigger rates untested.
Untested assumptions: gloves vs touch targets, attic noise vs speech recognition, HFP mic quality, battery drain.
