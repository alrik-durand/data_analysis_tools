import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="data_analysis_tools",
    packages=setuptools.find_packages(),
    version="0.0.3",
    license='GPLv3',
    author="Alrik Durand",
    author_email="alrik.durand@gmail.com",
    description="Tools to read and analyse data files and Qudi data files.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alrik-durand/data_analysis_tools",
    python_requires='>=3.6',
    install_requires=["numpy", "pandas", "matplotlib", "scipy", "lmfit"]
)