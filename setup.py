import os
from setuptools import setup

data_files = []
# Add all the templates
for (dirpath, dirnames, filenames) in os.walk('share/jupyter/'):
    if filenames:
        data_files.append((dirpath, [os.path.join(dirpath, filename) for filename in filenames]))

setup(
    data_files=data_files,
    include_package_data=True,
    keywords=[
        'ipython',
        'jupyter',
        'widgets',
        'voila'
    ],
)
