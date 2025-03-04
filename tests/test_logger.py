#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test module for PyOmnix logger functionality.
This file tests all functions and classes in the logger.py module.
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
import unittest
from typing import List, Optional

# Import the logger module
from PyOmnix.omnix_logger import (
    LoggerConfig,
    setup_logger,
    get_logger,
    close_logger,
    ExceptionHandler,
    validate,
    custom_excepthook
)


class TestLogger(unittest.TestCase):
    """Test cases for the logger module"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for log files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file_path = Path(self.temp_dir.name) / "test_log.log"
        
        # Store the original excepthook to restore it later
        self.original_excepthook = sys.excepthook
        
        # Store created loggers to clean them up later
        self.test_loggers = []

    def _close_all_handlers(self):
        """Close all handlers for all loggers to prevent file lock issues"""
        # First close handlers for loggers we explicitly created
        for logger_name in self.test_loggers:
            close_logger(logger_name)
                
        # Then check all loggers in the system
        for name, logger in logging.Logger.manager.loggerDict.items():
            if hasattr(logger, 'handlers'):
                for handler in list(logger.handlers):
                    if hasattr(handler, 'baseFilename') and str(self.log_file_path) in str(handler.baseFilename):
                        handler.close()
                        logger.removeHandler(handler)

    def tearDown(self):
        """Tear down test fixtures"""
        # Close all file handlers to prevent file lock issues
        self._close_all_handlers()
        
        # Clean up temporary directory
        self.temp_dir.cleanup()
        
        # Restore the original excepthook
        sys.excepthook = self.original_excepthook

    def read_log_file(self, file_path: Path) -> List[str]:
        """Helper method to read log file contents"""
        with open(file_path, 'r') as f:
            return f.readlines()

    def test_setup_logger_default(self):
        """Test setup_logger with default parameters"""
        logger = setup_logger()
        self.assertEqual(logger.name, LoggerConfig.DEFAULT_NAME)
        self.assertEqual(logger.level, LoggerConfig.DEFAULT_LEVEL)
        self.assertTrue(hasattr(logger, 'trace'))

    def test_setup_logger_custom(self):
        """Test setup_logger with custom parameters"""
        custom_name = "TestLogger"
        custom_level = logging.DEBUG
        
        logger = setup_logger(
            name=custom_name,
            log_level=custom_level,
            log_file=self.log_file_path,
            propagate=True
        )
        
        # Add to our list of loggers to clean up
        self.test_loggers.append(custom_name)
        
        self.assertEqual(logger.name, custom_name)
        self.assertEqual(logger.level, custom_level)
        self.assertTrue(logger.propagate)
        self.assertTrue(self.log_file_path.exists())

    def test_trace_level(self):
        """Test the custom TRACE log level"""
        logger_name = "TraceTest"
        logger = setup_logger(
            name=logger_name,
            log_level=LoggerConfig.TRACE,
            log_file=self.log_file_path
        )
        
        # Add to our list of loggers to clean up
        self.test_loggers.append(logger_name)
        
        trace_message = "This is a trace message"
        logger.trace(trace_message)
        
        log_contents = self.read_log_file(self.log_file_path)
        self.assertTrue(any(trace_message in line for line in log_contents))
        self.assertTrue(any("TRACE" in line for line in log_contents))

    def test_get_logger(self):
        """Test get_logger function"""
        # First create a logger
        test_name = "GetLoggerTest"
        setup_logger(name=test_name)
        
        # Then retrieve it
        retrieved_logger = get_logger(test_name)
        self.assertEqual(retrieved_logger.name, test_name)
        
        # Test getting the default logger
        default_logger = get_logger()
        self.assertEqual(default_logger.name, LoggerConfig.DEFAULT_NAME)

    def test_exception_handler(self):
        """Test ExceptionHandler class"""
        # Test get_log_level method
        self.assertEqual(
            ExceptionHandler.get_log_level(ValueError),
            logging.ERROR
        )
        
        # Test with an unmapped exception
        class CustomException(Exception):
            pass
        
        self.assertEqual(
            ExceptionHandler.get_log_level(CustomException),
            ExceptionHandler.DEFAULT_LEVEL
        )
        
        # Test format_exception method
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            formatted = ExceptionHandler.format_exception(exc_type, exc_value, exc_traceback)
            self.assertIsInstance(formatted, str)
            self.assertIn("ValueError: Test exception", formatted)

    def test_validate_function_pass(self):
        """Test validate function when condition passes"""
        # This should not raise an exception
        validate(True, "This should not raise an exception")

    def test_validate_function_fail(self):
        """Test validate function when condition fails"""
        # Setup a logger with a file to capture the error
        logger_name = "ValidateTest"
        setup_logger(
            name=logger_name,
            log_file=self.log_file_path
        )
        
        # Add to our list of loggers to clean up
        self.test_loggers.append(logger_name)
        
        error_message = "Validation failed"
        
        # This should raise an AssertionError
        with self.assertRaises(AssertionError):
            validate(
                False,
                error_message,
                logger_name=logger_name
            )
        
        # Check that the error was logged
        log_contents = self.read_log_file(self.log_file_path)
        self.assertTrue(any(error_message in line for line in log_contents))

    def test_custom_excepthook(self):
        """Test custom_excepthook function"""
        # Setup a logger with a file to capture the exception
        logger_name = "ExcepthookTest"
        setup_logger(
            name=logger_name,
            log_file=self.log_file_path
        )
        
        # Add to our list of loggers to clean up
        self.test_loggers.append(logger_name)
        
        # Set our custom excepthook
        sys.excepthook = custom_excepthook
        
        # Create a controlled exception
        try:
            # This will be caught by our excepthook
            raise KeyError("Test exception for excepthook")
        except KeyError:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            # Manually call the excepthook (normally Python would do this)
            custom_excepthook(exc_type, exc_value, exc_traceback)
        
        # Check that the exception was logged
        log_contents = self.read_log_file(self.log_file_path)
        self.assertTrue(any("KeyError: Test exception for excepthook" in line for line in log_contents))

    def test_close_logger(self):
        """Test close_logger function"""
        # Create a logger with a file handler
        logger_name = "CloseLoggerTest"
        logger = setup_logger(
            name=logger_name,
            log_file=self.log_file_path
        )
        
        # Add to our list of loggers to clean up
        self.test_loggers.append(logger_name)
        
        # Verify the logger has handlers
        self.assertTrue(len(logger.handlers) > 0)
        
        # Close the logger
        close_logger(logger_name)
        
        # Verify all handlers have been removed
        self.assertEqual(len(logger.handlers), 0)
        
        # Test closing the default logger
        default_logger = get_logger()
        self.assertTrue(len(default_logger.handlers) > 0)
        
        close_logger()
        
        # Verify all handlers have been removed from default logger
        self.assertEqual(len(default_logger.handlers), 0)


def run_manual_tests():
    """Run manual tests with visual output"""
    print("\n=== Running Manual Logger Tests ===\n")
    
    # Test basic logger setup
    print("Testing basic logger setup...")
    logger = setup_logger(name="ManualTest", log_level=logging.DEBUG)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    logger.trace("This is a trace message")
    print("Basic logger test complete\n")
    
    # Test logger with file output
    print("Testing logger with file output...")
    log_file = Path("manual_test.log")
    file_logger = setup_logger(
        name="FileLogger",
        log_file=log_file
    )
    file_logger.info("This message should be in the log file")
    print(f"Check {log_file} for log output")
    print("File logger test complete\n")
    
    # Test exception handling
    print("Testing exception handling...")
    try:
        print("Raising a ValueError...")
        raise ValueError("This is a test exception")
    except Exception:
        print("Exception caught and should be logged")
    print("Exception handling test complete\n")
    
    # Test validate function
    print("Testing validate function...")
    try:
        print("Validating a true condition (should pass)...")
        validate(True, "This should not be seen")
        print("Validation passed")
        
        print("Validating a false condition (should fail)...")
        validate(False, "This validation should fail")
    except AssertionError as e:
        print(f"Validation failed as expected with: {e}")
    print("Validate function test complete\n")
    
    print("=== Manual Tests Complete ===")


if __name__ == "__main__":
    # Run the automated tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Run manual tests with visual output
    run_manual_tests() 