# FieldCapture Xcode project

Open `FieldCapture.xcodeproj` on a Mac with Xcode 15 or newer. The project includes the current SwiftUI app, glasses connection scaffold, session state machine, Info.plist configuration, and Meta Wearables DAT package product references.

Before building: confirm the Meta DAT package version and product names against the current Meta SDK documentation, set the real bundle identifier and Meta App ID for your developer app, and complete the SDK verification points documented in `../FieldCapture/README.md`. `MetaAppID = 0` is only a development placeholder.

This project file has been generated but has not been compiled. The current Swift connection source contains API points explicitly marked `VERIFY`; adding an Xcode project does not prove device connectivity.
