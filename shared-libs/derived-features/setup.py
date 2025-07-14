from setuptools import setup, find_packages

setup(
    name="hss-derived-features",
    version="1.0.0",
    description="ChartMuse HSS Derived Features Library - Multimodal feature engineering",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)