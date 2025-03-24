from setuptools import setup, find_packages

setup(
    name="msds_chat_demo",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.31.0",
        "torch==2.2.0",
        "numpy>=1.24.0",
        "pillow==9.1.0",
        "pandas>=2.0.0",
        "transformers",
        "sentence-transformers",
        "accelerate>=0.26.0",
        "google-generativeai>=0.3.0",
        "scikit-learn>=1.0.2",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "pysqlite3-binary>=0.5.0",
        "pymongo[srv]",
        "dnspython",
    ],
) 