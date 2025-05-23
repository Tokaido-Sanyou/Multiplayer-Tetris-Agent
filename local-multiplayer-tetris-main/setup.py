from setuptools import setup, find_packages

setup(
    name="localMultiplayerTetris",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pygame>=2.0.0",
        "numpy>=1.19.0",
        "torch>=1.7.0",
    ],
    python_requires=">=3.7",
    author="Your Name",
    author_email="your.email@example.com",
    description="A local multiplayer Tetris game with reinforcement learning capabilities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/local-multiplayer-tetris",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 