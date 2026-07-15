from backend.services.session_manager import SessionManager


class FakeAssistant:
    pass


def test_document_ownership_is_session_scoped():
    manager = SessionManager(factory=FakeAssistant)
    manager.create_or_replace("user-a")
    manager.create_or_replace("user-b")
    manager.register_document("user-a", "record.pdf")
    assert manager.owns_document("user-a", "record.pdf")
    assert not manager.owns_document("user-b", "record.pdf")
