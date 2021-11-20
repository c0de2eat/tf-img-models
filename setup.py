import setuptools


setuptools.setup(
    name="tfim",
    version="0.0.1",
    author="c0de2eat",
    description="TensorFlow common codes for computer vision related tasks.",
    url="https://github.com/c0de2eat/tf-img-models",
    project_urls={
        "Bug Tracker": "https://github.com/c0de2eat/tf-img-models/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    install_requires=[
        "tensorflow_addons",
        "pydot",
    ],
    python_requires=">=3.8.8",
)
