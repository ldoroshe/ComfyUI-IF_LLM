"""Tests for send_request module logging configuration."""



class TestLoggingConfiguration:
    def test_logging_level_is_info(self):
        """Verify logging level is INFO (not DEBUG) in send_request.py."""
        import os

        send_request_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', 'send_request.py'
        )

        with open(send_request_path, 'r') as f:
            content = f.read()

        assert 'logging.basicConfig(level=logging.INFO' in content, \
            "Logging level should be INFO, not DEBUG for production"
        assert 'logging.basicConfig(level=logging.DEBUG' not in content, \
            "DEBUG level is too verbose for production"
