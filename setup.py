from setuptools import setup, find_packages

setup(
    name="fivegaps",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.25.2",
        "opencv-python>=4.8.0.76",
    ],
    author="Timo Kaiser",
    author_email="kaiser@tnt.uni-hannover.de",
    description="Code for 5GAPS-Demo Session",
    entry_points={
        'console_scripts': [
            'fivegaps-test = fivegaps.scripts.test:main',
            'fivegaps-speed-test = fivegaps.scripts.speed_test:main',
            # Add more scripts here if needed
        ],
    },
)
