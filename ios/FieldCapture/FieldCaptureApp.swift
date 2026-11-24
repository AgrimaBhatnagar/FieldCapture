import SwiftUI
import MWDATCore

@main
struct FieldCaptureApp: App {
    @StateObject private var glasses = GlassesConnection()

    init() {
        do { try Wearables.configure() }            // once, at launch
        catch { assertionFailure("Wearables.configure failed: \(error)") }
    }

    var body: some Scene {
        WindowGroup {
            ConnectView(glasses: glasses)
                .onOpenURL { glasses.handle(url: $0) }   // Meta AI calls back here after registration
                .task { glasses.observeRegistration() }
        }
    }
}

struct ConnectView: View {
    @ObservedObject var glasses: GlassesConnection
    var body: some View {
        VStack(spacing: 16) {
            Text("Session: \(glasses.machine.state.rawValue)").font(.title2.bold())
            Text("Registration: \(glasses.registration)")
            if let img = glasses.currentFrame { Image(uiImage: img).resizable().scaledToFit() }
            if let e = glasses.lastError { Text(e).foregroundStyle(.red) }
            HStack {
                Button("Register") { glasses.register() }
                Button("Start") { glasses.start() }
                Button("Stop") { glasses.stop() }
                Button("Capture") { glasses.capturePhoto() }.disabled(!glasses.machine.canCapture)
            }.buttonStyle(.borderedProminent)
        }.padding()
    }
}
