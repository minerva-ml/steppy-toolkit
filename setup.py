from setuptools import setup, find_packages

long_description = '''
Steppy-toolkit is complementary to the steppy library.

The goal of this package is to provide data scientist
with curated collection of highly parameterizable implementations of neural networks
together with a number of pre- and post-processing routines.

Steppy-toolkit offers implementations in popular frameworks, such as PyTorch, Keras and scikit-learn.

Steppy-toolkit is compatible with Python>=3.5
and is distributed under the MIT license.
'''

setup(name='steppy-toolkit',
      packages=find_packages(),
      version='0.1.9',
      description='Set of tools to make your work with steppy faster and more effective.',
      long_description=long_description,
      url='https://github.com/minerva-ml/steppy-toolkit',
      download_url='https://github.com/minerva-ml/steppy-toolkit/archive/0.1.9.tar.gz',
      author='Kamil A. Kaczmarek, Jakub Czakon',
      author_email='kamil.kaczmarek@neptune.ml, jakub.czakon@neptune.ml',
      keywords=['machine-learning', 'reproducibility', 'pipeline', 'tools'],
      license='MIT',
      install_requires=[
          'neptune-cli>=2.8.0',
          'setuptools>=39.2.0',
          'steppy>=0.1.9'],
      zip_safe=False,
      classifiers=[])
