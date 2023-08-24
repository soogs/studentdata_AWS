# exception.py #

# Let's write the exception...

import sys
import src.logger
import logging


# whenever an exception gets raised, this 
# should be pushed as the message
def error_message_detail(error, error_detail:sys):
	_,_,exc_tb = error_detail.exc_info()

	# the exc_tb gives where the exception,
	# where the error is..

	file_name = exc_tb.tb_frame.f_code.co_filename
	# my god, this is how to get the filename,
	# given in the exc_tb....

	error_message = "Error occured in python script name [{0}], line number [{1}], error message [{2}]".format(
		file_name, 
		exc_tb.tb_lineno,
		str(error)
		)

	return error_message


class CustomException(Exception):
	def __init__(self, error_message, error_detail:sys):
		super().__init__(error_message)
		self.error_message = error_message_detail(error_message, error_detail)


	def __str__(self):
		return self.error_message
		# this prints the message...


# trying to see if this works:
if __name__ == "__main__":

	try:
		a = 1 / 0 # creating an error
	except Exception as e:
		logging.info("Error: divide by zero.")
		raise CustomException(e, sys)