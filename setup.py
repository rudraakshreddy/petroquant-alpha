from setuptools import setup, find_packages

setup(
    name='crude-oil-crack-spreads',
    version='1.0.0',
    description='Mean Reversion Strategy for the 3:2:1 Crude Oil Crack Spread',
    author='Rudraaksh Reddy',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'statsmodels',
        'yfinance',
        'matplotlib',
        'seaborn',
        'plotly',
        'tqdm',
        'tabulate',
        'openpyxl'
    ],
)
