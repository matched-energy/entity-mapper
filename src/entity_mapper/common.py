class MappingException(Exception):
    """An REGO to BM mapping exception"""

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
