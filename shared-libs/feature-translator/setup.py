from setuptools import setup, find_packages

setup(
    name="hss-feature-translator",
    version="1.0.0",
    description="Feature translation library for Hit Song Science microservices",
    author="Hit Song Science Team",
    packages=find_packages(),
    package_data={
        "hss_feature_translator": ["feature_registry_v1.yaml"],
    },
    include_package_data=True,
    install_requires=[
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
)