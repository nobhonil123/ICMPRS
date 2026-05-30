"""
ICMPRS: Indian Context Multimodal Parkinson's Reference Standard
Setup file for pip-installable package.
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#") and not line.startswith("# ")
    ]

setup(
    name="icmprs",
    version="1.0.0",
    author="Nobhonil Roy Choudhury, Tamal Pal",
    author_email="nobhonilnew@gmail.com",
    description=(
        "Indian Context Multimodal Parkinson's Reference Standard — "
        "synthetic benchmark generator and CGMS classifier"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nobhonil123/ICMPRS",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "icmprs-generate=icmprs.generator:cli_generate",
            "icmprs-train=cgms.pipeline:cli_train",
            "icmprs-evaluate=evaluation.runner:cli_evaluate",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "Parkinson's disease",
        "multimodal screening",
        "synthetic data",
        "Indian populations",
        "Devanagari",
        "edge deployment",
    ],
)
