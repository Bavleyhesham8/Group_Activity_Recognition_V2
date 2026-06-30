from setuptools import setup, find_packages

setup(
    name="vgar",
    version="2.0.0",
    description="VAT-Former: Volleyball Actor-Team Transformer for Group Activity Recognition",
    author="Bavley Hesham",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "ultralytics>=8.0.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
)
