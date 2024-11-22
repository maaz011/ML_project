import sys
from src.logger import logging
## u can directly copy paste the exception handling code documentation from google BUT this is custom code
def error_message_detail(error,error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in pyhton script name [{0}] line number[{1}] error message [{2}]".format(file_name,exc_tb.tb_lineno,str(error))
    return error_message  
class CustomException(Exception):
    def __init__(self,error_message,error_details:sys): ## constructor
        super().__init__(error_message) ##inherit the init/exception class func,
        self.error_message=error_message_detail(error_message,error_details=error_details)
    
    def __str__(self):
        return self.error_message # raised error print 

# if __name__=="__main__":    ## //for print

#     try:
#         a=1/0
#     except Exception as e:
#         logging.info("Divide by zero error")
#         raise CustomException(e,sys)