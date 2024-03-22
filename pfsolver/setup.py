from setuptools import setup, find_packages

setup(
    name='pfsolver',
    version='0.1.0',
    author='Eric Kim',
    author_email='aegis4048@github.com',
    packages=find_packages(),
    description='A package for solving petroleum fractions',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[i.strip() for i in open("requirements.txt").readlines()],
    url='https://github.com/aegis4048/PetroleumFractionSolver',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)