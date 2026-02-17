import os
import tempfile


class TestSetupLogger:
    """Tests for logger.py setup_logger function."""

    def test_setup_logger_creates_log_directory(self):
        """Test that setup_logger creates the log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "subdir", "test.log")

            from src.utils.logger import setup_logger

            logger = setup_logger(log_file=log_file)

            assert os.path.exists(os.path.dirname(log_file))
            assert logger is not None

    def test_setup_logger_returns_logger_instance(self):
        """Test that setup_logger returns a logger instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")

            from src.utils.logger import setup_logger

            logger = setup_logger(log_file=log_file)

            assert logger is not None
            # Verify it's a loguru logger by checking it has expected methods
            assert hasattr(logger, "info")
            assert hasattr(logger, "error")
            assert hasattr(logger, "warning")
            assert hasattr(logger, "debug")

    def test_setup_logger_with_custom_parameters(self):
        """Test setup_logger with custom level, log_file, and rotation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "custom", "app.log")

            from src.utils.logger import setup_logger

            logger = setup_logger(level="DEBUG", log_file=log_file, rotation="5 MB")

            assert logger is not None
            assert os.path.exists(os.path.dirname(log_file))

            # Write a test message to verify logger works
            logger.debug("Test debug message")
            logger.info("Test info message")

            # Verify log file was created and contains messages
            assert os.path.exists(log_file)
            with open(log_file) as f:
                content = f.read()
                assert "Test debug message" in content
                assert "Test info message" in content

    def test_setup_logger_default_parameters(self):
        """Test setup_logger with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp directory to avoid creating logs in project root
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                from src.utils.logger import setup_logger

                logger = setup_logger()

                assert logger is not None
                # Default log directory should be created
                assert os.path.exists("logs")

                # Write a message and verify it appears in the log
                logger.info("Test message")
                assert os.path.exists("logs/crypto_bot.log")

            finally:
                os.chdir(original_cwd)

    def test_setup_logger_handles_existing_directory(self):
        """Test that setup_logger works when log directory already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs")
            os.makedirs(log_dir)
            log_file = os.path.join(log_dir, "test.log")

            from src.utils.logger import setup_logger

            # Should not raise an error even though directory exists
            logger = setup_logger(log_file=log_file)

            assert logger is not None
            assert os.path.exists(log_dir)

    def test_setup_logger_rotation_parameter(self):
        """Test that rotation parameter is accepted without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "rotated.log")

            from src.utils.logger import setup_logger

            # Test different rotation values
            logger1 = setup_logger(log_file=log_file, rotation="1 MB")
            assert logger1 is not None

            logger2 = setup_logger(log_file=log_file, rotation="100 KB")
            assert logger2 is not None

            logger3 = setup_logger(log_file=log_file, rotation="1 day")
            assert logger3 is not None
