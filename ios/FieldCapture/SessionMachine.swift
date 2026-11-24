// SDK-independent session state machine. Driven by DeviceSessionState events from GlassesConnection.
// Pure Swift, no SDK imports, so it can be unit-tested without glasses or Xcode UI.
import Foundation

enum SessionState: String { case idle, starting, live, paused, dropped, stopped }
enum SessionEvent { case start, sdkStarted, sdkPaused, sdkStopped, userStop, startFailed }

struct SessionMachine {
    private(set) var state: SessionState = .idle
    private(set) var log: [String] = []
    private(set) var hadLive = false

    mutating func handle(_ e: SessionEvent) {
        let from = state
        switch (state, e) {
        case (.idle, .start), (.stopped, .start), (.dropped, .start): state = .starting
        case (.starting, .sdkStarted): state = .live; hadLive = true
        case (.starting, .startFailed): state = hadLive ? .dropped : .idle
        case (.live, .sdkPaused): state = .paused
        case (.paused, .sdkStarted): state = .live
        case (.live, .sdkStopped), (.paused, .sdkStopped), (.starting, .sdkStopped): state = .dropped
        case (_, .userStop): state = .stopped
        default: break   // ignore illegal transitions
        }
        log.append("\(from.rawValue) -> \(state.rawValue) on \(e)")
    }
    var canCapture: Bool { state == .live }
    var shouldReconnect: Bool { state == .dropped }
}
