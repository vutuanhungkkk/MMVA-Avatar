from backend.services.session_manager import SessionManager


class FakeAssistant:
    pass


def test_sessions_are_isolated():
    manager = SessionManager(factory=FakeAssistant)
    first = manager.create_or_replace("user-a")
    second = manager.create_or_replace("user-b")
    assert first is manager.get("user-a")
    assert second is manager.get("user-b")
    assert first is not second


def test_replacing_one_session_does_not_touch_another():
    manager = SessionManager(factory=FakeAssistant)
    old = manager.create_or_replace("user-a")
    other = manager.create_or_replace("user-b")
    assert manager.create_or_replace("user-a") is not old
    assert manager.get("user-b") is other
