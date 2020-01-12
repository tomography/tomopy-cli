from setuptools import setup, find_packages

setup(
    name='tomopy-cli',
    version=open('VERSION').read().strip(),
    #version=__version__,
    author='Francesco De Carlo',
    author_email='decarlof@gmail.com',
    url='https://github.com/decarlof/tomopy-cli',
    packages=find_packages(),
    scripts=['bin/tomopy'],
    description='cli for tomopy',
    zip_safe=False,
)

