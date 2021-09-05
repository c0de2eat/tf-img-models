import setuptools


setuptools.setup(
    name="tfim",
    version="0.0.1",
    author="xingzhaolee",
    description="TensorFlow common codes for computer vision related tasks.",
    url="https://github.com/xingzhaolee/tf-img-models",
    project_urls={
        "Bug Tracker": "https://github.com/xingzhaolee/tf-img-models/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    install_requires=[
        "cython",
        "tensorflow_datasets",
        "tensorflow_addons",
        "matplotlib",
        "opencv-python",
        "pydot",
        "tqdm",
    ],
    python_requires=">=3.6.9",
)
