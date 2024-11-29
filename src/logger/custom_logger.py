import logging


class CustomLogger(logging.Logger):
    def __init__(self, name: str, level: int | str = 0) -> None:
        super().__init__(name, level)
    
    def error(self, msg: object, *args: object, exc_info: Exception, **kwargs) -> None:
        
        # Extract file name and line no. where error occurred
        file_name = exc_info.__traceback__.tb_frame.f_code.co_filename
        line_no = exc_info.__traceback__.tb_lineno
        
        # Create detailed error message
        detailed_message = (
            f"Type: {type(exc_info).__name__}, Location: {file_name}:{line_no}, Message: {msg}, Root Message: {str(exc_info)}"
        )
        
        # Log the error
        return super().error(detailed_message, *args, exc_info=exc_info, **kwargs)