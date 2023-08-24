# setup.py

# setup.py is responsible is responsible for creating a 
# ML application as a package in itself

# building a ML package as a packge!

from setuptools import find_packages, setup 
# find_packages finds out the packages in the 
# ML project that we are working on


HYPEN_E_DOT='-e .'
# for treating the -e. thing in the requirements.txt

def get_requirements(path:str):
	'''
	This function returns the list of requirements from a text file
	'''
	requirements = []

	with open(path) as file_obj:
		requirements = file_obj.readlines()
		requirements = [req.replace("\n", "") for req in requirements]

	if HYPEN_E_DOT in requirements:
		requirements.remove(HYPEN_E_DOT)

	return requirements

# the following basically specifies the "metadata"
setup(

	name = "MLproject",
	version = "0.0.1",
	author = "soogs",
	author_email = "s.park_1@tilburguniversity.edu",
	packages = find_packages(),
	install_requires = get_requirements("requirements.txt")

	)

# get_requirements is a user-defined function
# for fetching the packages mentioned in the "requirements.txt"


