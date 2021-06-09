from setuptools import setup, find_packages
from setuptools.command.install import install
import os

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):        
        install.run(self)
        from tomopy_cli.auto_complete import create_complete_tomopy        
        import pathlib
        create_complete_tomopy.run(str(pathlib.Path.home())+'/complete_tomopy.sh')
        print('For autocomplete please run: \n\n $ source '+str(pathlib.Path.home())+'/complete_tomopy.sh\n'     )

setup(
    name='tomopy-cli',
    version=open('VERSION').read().strip(),
    #version=__version__,
    author='Francesco De Carlo',
    author_email='decarlof@gmail.com',
    url='https://github.com/decarlof/tomopy-cli',
    packages=find_packages(),
    include_package_data = True,
    scripts=['bin/tomopy.py'],  
    entry_points={'console_scripts':['tomopy = tomopy:main'],},
    description='cli for tomopy',
    zip_safe=False,
    cmdclass={'install': PostInstallCommand},
)

