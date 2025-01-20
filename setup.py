import setuptools

with open("README.md", "r") as readme:
    long_description = readme.read()

with open("requirements.txt", "r") as req:
    requirements = req.read().splitlines()

setuptools.setup(
    name="hierarchical_st_nav_aif",
    version="0.2.0",
    author="DML group",
    author_email="",
    description=" D Hierarchical active inference experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 4 - Beta"
    ],
    python_requires=">3.9.7",
    install_requires=[
        'numpy>=1.17.4',
        'pynput>=1.6.8',
        'torch>=1.10.2',
        'matplotlib>=3.1.2',
        'utils>=1.0',
        'torchvision>0.11.3',
        'tqdm>=4.63.0',
        'gym==0.17.0',
        'dill>=0.3.6',
        'imageio>=2.16.1',
        'Pillow>=9.4.0',
        'pyquaternion>=0.9.9',
        'PyYAML>=5.1.2',
        'pandas>=1.4.1',
        'seaborn>=0.12.2',
        'setuptools>=45.2.0',
        'h5py>=2.6.0'
    ],

)