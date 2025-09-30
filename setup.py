from setuptools import setup, find_packages

setup(
    name="TORCH",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.5"
    ],
    python_requires=">=3.7, <3.10",  # 指定 Python 版本
    description="Tri-block Operator splitting for Resistant Composite Hypothesis (TORCH) solver package for Compound Empirical Likelihood (CEL).",
    author="Zhaojun Hu",
    url="https://github.com/Hypon-Jun/TORCH-for-CEL",
)