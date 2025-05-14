#!/usr/bin/env python3
"""
Test runner script for the Statista Context API
Run this script to execute all tests for the application
"""

import sys
import pytest


def main():
    """Run pytest on all test files"""
    print("Running Statista Context API tests...")
    
    # Run all tests in the tests directory
    exit_code = pytest.main(["-v", "app/tests/"])
    
    # Exit with the pytest exit code
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 