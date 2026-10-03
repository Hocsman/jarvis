"""
Tests for the face window: where it sits, and that it shows what the
assistant is doing.
"""

import os
from unittest.mock import patch, MagicMock
import pytest


class TestFaceWindowPositioning:
    """Tests for FaceWindow positioning on the right side of screen."""

    def test_positions_on_right_side_of_screen(self):
        """FaceWindow should position itself on the right side of the screen."""
        # Mock screen geometry
        mock_screen = MagicMock()
        mock_screen.availableGeometry.return_value = MagicMock(
            right=lambda: 1920,
            top=lambda: 0,
            height=lambda: 1080,
        )

        # Mock QApplication.primaryScreen
        with patch(
            "desktop_app.face_widget.QApplication.primaryScreen", return_value=mock_screen
        ):
            # Import after patching to avoid needing actual display
            from desktop_app.face_widget import FaceWindow

            # Mock the parent class __init__ to avoid Qt initialization issues
            with patch.object(FaceWindow, "__init__", lambda self, parent=None: None):
                window = FaceWindow.__new__(FaceWindow)
                window._width = 350
                window._height = 450

                # Mock width() and height() methods
                window.width = lambda: 350
                window.height = lambda: 450

                # Track move calls
                move_calls = []
                window.move = lambda x, y: move_calls.append((x, y))

                # Call the positioning method
                window._position_on_right()

                # Verify positioning
                assert len(move_calls) == 1
                x, y = move_calls[0]

                # Should be on right side with 20px margin
                # x = 1920 - 350 - 20 = 1550
                assert x == 1550

                # Should be vertically centered
                # y = 0 + (1080 - 450) // 2 = 315
                assert y == 315

    def test_handles_none_screen_gracefully(self):
        """FaceWindow should handle missing screen gracefully."""
        with patch(
            "desktop_app.face_widget.QApplication.primaryScreen", return_value=None
        ):
            from desktop_app.face_widget import FaceWindow

            with patch.object(FaceWindow, "__init__", lambda self, parent=None: None):
                window = FaceWindow.__new__(FaceWindow)

                move_calls = []
                window.move = lambda x, y: move_calls.append((x, y))

                # Should not raise an exception
                window._position_on_right()

                # Should not move if no screen
                assert len(move_calls) == 0

    def test_adapts_to_different_screen_sizes(self):
        """FaceWindow should adapt to different screen sizes."""
        test_cases = [
            # (screen_right, screen_top, screen_height, expected_x, expected_y)
            (1920, 0, 1080, 1550, 315),  # Standard 1080p
            (2560, 0, 1440, 2190, 495),  # 1440p
            (3840, 0, 2160, 3470, 855),  # 4K
            (1366, 0, 768, 996, 159),  # Common laptop
        ]

        window_width = 350
        window_height = 450
        margin = 20

        for screen_right, screen_top, screen_height, expected_x, expected_y in test_cases:
            mock_screen = MagicMock()
            mock_screen.availableGeometry.return_value = MagicMock(
                right=lambda r=screen_right: r,
                top=lambda t=screen_top: t,
                height=lambda h=screen_height: h,
            )

            with patch(
                "desktop_app.face_widget.QApplication.primaryScreen",
                return_value=mock_screen,
            ):
                from desktop_app.face_widget import FaceWindow

                with patch.object(
                    FaceWindow, "__init__", lambda self, parent=None: None
                ):
                    window = FaceWindow.__new__(FaceWindow)
                    window.width = lambda: window_width
                    window.height = lambda: window_height

                    move_calls = []
                    window.move = lambda x, y: move_calls.append((x, y))

                    window._position_on_right()

                    assert len(move_calls) == 1
                    x, y = move_calls[0]
                    assert x == expected_x, f"For screen {screen_right}x{screen_height}"
                    assert y == expected_y, f"For screen {screen_right}x{screen_height}"


@pytest.fixture
def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


class TestFaceFollowsTheSharedState:
    """The face reads the assistant's state from ``jarvis.state`` each frame,
    so it shows what the voice pipeline published whether the daemon runs in
    this process, in a subprocess, or not yet."""

    @pytest.mark.unit
    def test_a_new_face_is_asleep(self, _qapp):
        from desktop_app.face_widget import LowPolyFaceWidget
        from jarvis.state import JarvisState

        face = LowPolyFaceWidget()
        face._animate()

        assert face._jarvis_state == JarvisState.ASLEEP

    @pytest.mark.unit
    @pytest.mark.parametrize("state_name", ["IDLE", "LISTENING", "THINKING", "SPEAKING", "DICTATING"])
    def test_the_face_shows_what_the_pipeline_published(self, _qapp, state_name):
        from desktop_app.face_widget import LowPolyFaceWidget
        from jarvis.state import JarvisState, get_jarvis_state

        face = LowPolyFaceWidget()
        state = JarvisState[state_name]

        get_jarvis_state().set_state(state)
        face._animate()

        assert face._jarvis_state == state

    @pytest.mark.unit
    def test_the_state_has_one_home_and_the_face_does_not_keep_a_copy(self):
        """The enum is the core's. A second definition in the desktop package
        would give the face states the pipeline never publishes."""
        from desktop_app import face_widget
        from jarvis import state as core_state

        assert face_widget.JarvisState is core_state.JarvisState
        assert face_widget.get_jarvis_state is core_state.get_jarvis_state
