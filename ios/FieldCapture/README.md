# iOS connection layer (UNCOMPILED)
Needs a Mac with Xcode 15+ (SDK guide) and iOS 16+ target. Steps:
1. Install Meta's skills first: `claude plugin install mwdat-ios@mwdat-ios-marketplace` (or read AGENTS.md in the SDK repo).
2. New SwiftUI app. File > Add Package Dependencies > https://github.com/facebook/meta-wearables-dat-ios ; add MWDATCore + MWDATCamera.
3. Merge Info.plist.fragment.xml. Add these three .swift files. Enable Developer Mode in the Meta AI app (tap App version 5 times).
4. For no-glasses testing, build Meta's CameraAccess sample in Debug: it has a built-in Mock Device Kit panel (phone camera or h265 video file feed).
   Mock Device Kit does not support Display glasses.
5. Build. Fix the lines marked VERIFY. Then run: Register -> Start -> Capture; hinge-close / take glasses off to test pause and drop.
Known unknowns: photoDataPublisher location, PermissionStatus case names, listener token type, resolution needed for nameplate serials,
HFP mic quality (speech uses the iPhone mic path via SFSpeech, not wired here). Frame-to-quality-gate and voice parser ports are not yet written in Swift.
