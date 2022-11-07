"""
Loads versions from the__versions__.py module. Change version in
that module, and it will be automatically populated here.
"""
import setuptools
import os

here = os.path.abspath(os.path.dirname(__file__))
version_path = os.path.join(here, "prompts", "__version__.py")
about = {}  # type: dict

with open(version_path, "r") as f:
    exec(f.read(), about)


setuptools.setup(
    name="prompts",
    version=about["__version__"],
    description="This project aims to evaluate LLM performance",
    packages=setuptools.find_packages(include=["prompts*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
)
