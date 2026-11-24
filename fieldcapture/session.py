"""DAT 1.0 session controller with Mock Device Kit and phone audio fallback."""
from dataclasses import dataclass, field


@dataclass
class SessionController:
    job_id: str
    started_at: str
    device_mode: str = "mock_device_kit"
    audio_fallback_reason: str | None = None
    state: str = "idle"
    connected: bool = True
    drop_recoveries: int = 0
    events: list[str] = field(default_factory=list)

    def register_and_grant_permissions(self, audio_available: bool = True):
        if not audio_available:
            self.audio_fallback_reason = (
                "Mock Device Kit audio path unavailable; falling back to phone microphone."
            )
        self.events.append("permissions_granted")

    def start(self):
        self.state = "active"
        self.connected = True
        self.events.append("start")

    def pause(self):
        if self.state == "active":
            self.state = "paused"
            self.events.append("pause")

    def resume(self):
        if self.state == "paused":
            self.state = "active"
            self.events.append("resume")

    def simulate_disconnect(self):
        self.connected = False
        self.events.append("disconnect")

    def recover_after_disconnect(self):
        if not self.connected:
            self.connected = True
            self.state = "active"
            self.drop_recoveries += 1
            self.events.append("recover")

    def stop(self):
        self.state = "stopped"
        self.events.append("stop")
