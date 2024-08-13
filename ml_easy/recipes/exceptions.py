import json

from ml_easy.recipes.enum import MLFlowErrorCode


class MlflowException(Exception):
    """
    Generic exception thrown to surface failure information about external-facing operations.
    The error message associated with this exception may be exposed to clients in HTTP responses
    for debugging purposes. If the error text is sensitive, raise a generic `Exception` object
    instead.
    """

    def __init__(self, message: str, error_code: MLFlowErrorCode, **kwargs):
        """
        Args:
            message: The message or exception describing the error that occurred. This will be
                included in the exception's serialized JSON representation.
            error_code: An appropriate error code for the error that occurred; it will be
                included in the exception's serialized JSON representation. This should
                be one of the codes listed in the `recipes.enum.MLFlowErrorCode` enum.
            kwargs: Additional key-value pairs to include in the serialized JSON representation
                of the MlflowException.
        """
        self.error_code: MLFlowErrorCode = error_code
        message = str(message)
        self.message = message
        self.json_kwargs = kwargs
        super().__init__(message)

    def serialize_as_json(self) -> str:
        exception_dict = {'error_code': self.error_code, 'message': self.message}
        exception_dict.update(self.json_kwargs)
        return json.dumps(exception_dict)
