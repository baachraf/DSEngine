from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith('#')]

setup(
    name='ds_engine',
    version='1.0.0',
    author='DSEngine Contributors',
    description='A declarative data science and EDA library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=requirements,
    extras_require={
        'docs': ['sphinx', 'sphinx-rtd-theme', 'nbsphinx'],
        'dev':  ['pytest', 'pytest-cov'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
