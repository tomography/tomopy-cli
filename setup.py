from setuptools import setup, find_packages
from setuptools.command.install import install
import os

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):        
        install.run(self)
        # print('Add autocomplete for tomopy recon')        
        # os.system('source ./tomopy_cli/auto_complete/complete_tomopy.sh')
        # print('Autocomplete done')        

setup(
    name='tomopy-cli',
    version=open('VERSION').read().strip(),
    #version=__version__,
    author='Francesco De Carlo',
    author_email='decarlof@gmail.com',
    url='https://github.com/decarlof/tomopy-cli',
    packages=find_packages(),
    include_package_data = True,
    scripts=['bin/tomopy'],
    description='cli for tomopy',
    zip_safe=False,
    cmdclass={'install': PostInstallCommand},
)

