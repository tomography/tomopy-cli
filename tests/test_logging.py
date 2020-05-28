import unittest
import io
import logging
import sys
import os

from tomopy_cli import log as tplogging


# Set up variables to capture logging output

class LogFormattingTests(unittest.TestCase):
    temp_logfile = 'tests-temp.log'
    
    def setUp(self):
        self.log_capture = io.StringIO()
        ch = logging.StreamHandler(self.log_capture)
        tplogging.logger.addHandler(ch)
    
    def tearDown(self):
        # Remove temporary log files
        if os.path.exists(self.temp_logfile):
            os.remove(self.temp_logfile)
    
    def test_setup_custom_logger(self):
        """Verify that setting up a custom logger produces formatted output."""
        # Capture stderr to a string
        stderr_io = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = stderr_io
        # Do some logging both with and without setting up custom logger
        try:
            tplogging.warning("No danger, Will Robinson")
            tplogging.setup_custom_logger(lfname=None)
            tplogging.warning("Formatted danger, Will Robinson")
        finally:
            # Restore stderr
            sys.stderr = old_stderr
        # Read the logged output to check for correctness
        stderr_io.seek(0)
        loglines = stderr_io.readlines()
        self.assertEqual(len(loglines), 1)
        self.assertNotIn("No danger", loglines[0])
        self.assertIn("\x1b[33m", loglines[0])
        self.assertIn("\x1b[0m", loglines[0])
    
    def test_colored_log_formatter(self):
        # Logging without colored formatting
        ch = logging.StreamHandler(self.log_capture)
        logger = logging.getLogger('_test_logger')
        logger.addHandler(ch)
        logger.warning("Danger, Will Robinson")
        # Logging with color formatting
        ch.setFormatter(tplogging.ColoredLogFormatter('%(asctime)s - %(message)s'))
        logger.warning("Formatted danger, Will Robinson")
        # Check the logged output for formatting
        self.log_capture.seek(0)
        loglines = self.log_capture.readlines()
        self.assertNotIn("\x1b[33m", loglines[0])
        self.assertNotIn("\x1b[0m", loglines[0])
        self.assertIn("\x1b[33m", loglines[1])
        self.assertIn("\x1b[0m", loglines[1])
