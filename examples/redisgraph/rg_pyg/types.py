from enum import Enum


class Type(Enum):
    """The type of a value in the database."""
    BOOL = "BOOL"
    INT64 = "INT64"
    FLOAT = "FLOAT"
    STRING = "STRING"
    DOUBLE = "DOUBLE"

