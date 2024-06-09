from setuptools import setup, find_packages

setup(
    name='AprendizajeRefuerzUS',
    version='0.1',
    packages=find_packages(include=['AprendizajeRefuerzUS', 'AprendizajeRefuerzUS.*']),
    description='Una aproximación al aprendizaje por refuerzo.',
    author='Ramon Gavira Sánchez y Daniel Ruiz López',
    author_email1='ramgavsan@alum.us.es',
    author_email2='danruilop1@alum.us.es',
    url='https://github.com/rgavira123/robot-aprendizaje-refuerzo',
    install_requires=[
        'ipykernel',

        'pymdptoolbox',

        'matplotlib',

        'numpy',
    ],

    include_package_data=True,
    package_data={
        'AprendizajeRefuerzUS': ['maps/*.txt'],
    },
    
)