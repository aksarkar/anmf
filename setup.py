import setuptools

setuptools.setup(
  name='anmf',
  description='Amortized NMF',
  version='0.1',
  url='https://www.github.com/aksarkar/anmf',
  author='Abhishek Sarkar',
  author_email='aksarkar@uchicago.edu',
  license='MIT',
  install_requires=[
    'numpy',
    'scipy',
    'torch',
  ],
  packages=setuptools.find_packages('src'),
  package_dir={'': 'src'},
)
