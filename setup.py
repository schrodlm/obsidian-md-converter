from setuptools import setup, find_packages

setup(
    name='obsidian-md-converter',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['PyYAML>=6.0'],
    extras_require={
        'test': ['pytest>=7.0'],
    },
    entry_points={
        'console_scripts': [
            'obsidian-md-converter=obsidian_md_converter.cli:main',
        ],
    },
    python_requires='>=3.10',
)
