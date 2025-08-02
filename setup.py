from setuptools import setup, find_packages

setup(
    name='kaggle_ml_pipeline',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost',
        'category_encoders', 'matplotlib', 'seaborn', 'wandb', 'shap'
    ],
    entry_points={
        'console_scripts': [
            'run-pipeline=main:run_pipeline'
        ]
    },
    author='Your Name',
    description='End-to-end Kaggle ML pipeline with ensembling, SHAP, and W&B',
    license='MIT'
)
