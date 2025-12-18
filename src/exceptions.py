# start src/exceptions.py
"""Custom exceptions for the LinkedIn Bias Auditor.

This module defines project-specific exceptions that provide clearer
error handling and better debugging information than generic exceptions.
"""


class BiasAuditError(Exception):
    """Base exception for all bias audit operations.

    All custom exceptions in this project inherit from this class,
    allowing callers to catch all project-specific errors with a
    single except clause if desired.
    """

    pass


class ConfigurationError(BiasAuditError):
    """Raised when configuration is invalid or incomplete.

    Examples:
        - Missing required configuration file
        - Invalid configuration values
        - Missing required configuration keys
    """

    pass


class DataValidationError(BiasAuditError):
    """Raised when input data fails validation.

    Examples:
        - Malformed JSON in input files
        - Missing required fields in data records
        - Mismatched data between male and female datasets
        - Invalid data types in records
    """

    pass


class APIConnectionError(BiasAuditError):
    """Raised when connection to LM Studio API fails.

    Examples:
        - LM Studio server not running
        - Network connectivity issues
        - API timeout exceeded
        - Invalid API response format
    """

    pass


class EmbeddingError(BiasAuditError):
    """Raised when embedding generation fails.

    Examples:
        - Invalid embedding response structure
        - Empty embedding returned
        - Embedding dimension mismatch
    """

    pass


class DatabaseError(BiasAuditError):
    """Raised when database operations fail.

    Examples:
        - Failed to initialize database
        - Failed to save audit results
        - Database connection lost
    """

    pass


# end src/exceptions.py
