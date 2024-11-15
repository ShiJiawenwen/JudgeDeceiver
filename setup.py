import os

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            return line.split("'")[1]

    raise RuntimeError('Unable to find version string.')


with open('requirements.txt', 'r') as requirements:
    setup(name='judge_attack',
          version=get_version('judge_attack/__init__.py'),
          install_requires=list(requirements.read().splitlines()),
          packages=find_packages(),
          description='library for prompt injection to llm as a judge',
          python_requires='>=3.6',
          author='Jiawen Shi',
          author_email='shijiawen1223@gmail.com',
          classifiers=[
              'Programming Language :: Python :: 3',
              'License :: OSI Approved :: MIT License',
              'Operating System :: OS Independent'
          ],
          long_description=long_description,
          long_description_content_type='text/markdown')