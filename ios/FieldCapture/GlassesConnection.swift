// Connects FieldCapture to Meta AI glasses (or Mock Device Kit) via the Device Access Toolkit 1.0.
// STATUS: written from Meta's official iOS skill files and docs (getting-started, session-lifecycle,
// build-integration-ios). NOT COMPILED OR RUN: no Xcode in the build sandbox. Lines marked VERIFY are
// where the docs were ambiguous or inconsistent. Fix against Xcode's autocomplete / the CameraAccess sample.
import SwiftUI
import MWDATCore
import MWDATCamera

@MainActor
final class GlassesConnection: ObservableObject {
    @Published private(set) var machine = SessionMachine()
    @Published private(set) var registration = "unknown"
    @Published private(set) var currentFrame: UIImage?
    @Published private(set) var lastError: String?

    /// Called with each JPEG photo captured from the glasses camera.
    var onPhoto: ((Data) -> Void)?

    private let wearables = Wearables.shared
    private var session: DeviceSession?
    private var stream: MWDATCamera.Stream?          // qualified: Foundation also has a `Stream`
    private var listenerTokens: [Any] = []           // keep listeners alive; VERIFY token type
    private var userStopped = false
    private var reconnectAttempt = 0
    private var reconnectTask: Task<Void, Never>?

    // MARK: Registration (once per install; Meta AI app opens and calls back via URL)
    func observeRegistration() {
        Task { for await s in wearables.registrationStateStream() { registration = "\(s)" } }
    }
    func register() {
        Task { do { try await wearables.startRegistration() } catch { fail("registration: \(error)") } }
    }
    func handle(url: URL) {
        Task { _ = try? await wearables.handleUrl(url) }
    }

    // MARK: Permissions. A device does not appear until a permission is granted in Meta AI.
    private func ensureCameraPermission() async throws -> Bool {
        var status = try await wearables.checkPermissionStatus(.camera)
        if status != .granted { status = try await wearables.requestPermission(.camera) }  // VERIFY: `.granted` case name
        return status == .granted
    }

    // MARK: Session lifecycle
    func start() {
        userStopped = false
        reconnectAttempt = 0
        beginSession()
    }

    func stop() {
        userStopped = true
        reconnectTask?.cancel()
        session?.stop()                 // fire-and-forget; STOPPED arrives on stateStream
        machine.handle(.userStop)
    }

    private func beginSession() {
        machine.handle(.start)
        Task {
            do {
                guard try await ensureCameraPermission() else { fail("camera permission denied"); machine.handle(.startFailed); return }
                // STOPPED is terminal: always create a NEW session (docs: opening hinges does not restart sessions).
                let s = try wearables.createSession(deviceSelector: AutoDeviceSelector(wearables: wearables))
                session = s
                try s.start()
                for await state in s.stateStream() {
                    switch state {
                    case .started:
                        if stream == nil { try attachCamera(to: s) }
                        reconnectAttempt = 0
                        machine.handle(.sdkStarted)
                    case .paused:
                        machine.handle(.sdkPaused)          // keep connection, wait for started/stopped; do not restart
                    case .stopped:
                        releaseStream()
                        machine.handle(userStopped ? .userStop : .sdkStopped)
                        if machine.shouldReconnect { scheduleReconnect() }
                    default: break
                    }
                }
            } catch {
                fail("session: \(error)")
                machine.handle(.startFailed)
                if machine.shouldReconnect { scheduleReconnect() }
            }
        }
    }

    // MARK: Drop recovery: exponential backoff, new session each time. Outbox keeps captures meanwhile.
    private func scheduleReconnect() {
        guard !userStopped, reconnectAttempt < 6 else { return }
        reconnectAttempt += 1
        let delay = min(30.0, pow(2.0, Double(reconnectAttempt)))
        reconnectTask?.cancel()
        reconnectTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            guard let self, !Task.isCancelled, !self.userStopped, self.machine.shouldReconnect else { return }
            self.beginSession()
        }
    }

    // MARK: Camera. Must be added AFTER the session reaches .started.
    private func attachCamera(to s: DeviceSession) throws {
        // VERIFY resolution: docs sample uses .low; nameplate serials may need higher. Measure in your eval.
        let config = StreamConfiguration(videoCodec: .raw, resolution: .low, frameRate: 24)
        guard let camera = try s.addCamera(config: config) else { fail("addCamera returned nil"); return }
        let st = camera.stream
        stream = st
        listenerTokens.append(st.videoFramePublisher.listen { [weak self] frame in
            guard let image = frame.makeUIImage() else { return }
            Task { @MainActor in self?.currentFrame = image }
        })
        // VERIFY: docs show photoDataPublisher on the stream session; confirm it is on `Stream` in 1.0.
        listenerTokens.append(st.photoDataPublisher.listen { [weak self] photo in
            let data = photo.data
            Task { @MainActor in self?.onPhoto?(data) }
        })
        st.start()
    }

    private func releaseStream() { listenerTokens.removeAll(); stream = nil }

    /// Ask the glasses for a still. Result arrives on onPhoto. Returns false if the stream is not live.
    @discardableResult
    func capturePhoto() -> Bool {
        guard machine.canCapture, let st = stream else { return false }
        return st.capturePhoto(format: .jpeg)
    }

    private func fail(_ m: String) { lastError = m }
}
