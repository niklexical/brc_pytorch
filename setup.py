"""Install package."""
from setuptools import setup, find_packages

setup(
    name='brc_pytorch',
    version='0.0.1',
    description=('Pytorch Implementation of BRC.'),
    long_description=open('README.md').read(),
    url='https://github.com/niklexical/brc_pytorch',
    author='Nikita Janakarajan, Jannis Born',
    author_email=('nikita.janakarajan907@gmail.com, jannis.born@gmx.de'),
    install_requires=['numpy', 'pandas', 'scipy', 'torch'],
    packages=find_packages('.'),
    zip_safe=False,
)
