"""Install package."""
import os
from setuptools import setup, find_packages

if os.path.exists('README.md'):
    long_description = open('README.md').read()
else:
    long_description = '''Pytorch Implementation of Bistable Recurrent Cell.'''

setup(
    name='brc_pytorch',
    version='0.1.3',
    license='MIT',
    description=('Pytorch Implementation of BRC.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/niklexical/brc_pytorch',
    author='Nikita Janakarajan, Jannis Born',
    author_email=('nikita.janakarajan907@gmail.com, jannis.born@gmx.de'),
    install_requires=['numpy', 'torch'],
    keywords=['PyTorch', 'Deep Learning', 'RNN', 'BRC'],
    packages=find_packages('.'),
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
