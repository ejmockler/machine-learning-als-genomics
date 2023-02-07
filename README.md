# Stacking ensemble for modeling ALS genomics 
[pending data access]

## Tools
- Python >=3.9
- Workflow: [Prefect](https://www.prefect.io)
- Modeling: [scikit-learn](https://scikit-learn.org/)
- Case/control dataset management: [tidyML](https://pypi.org/project/tidyML/)
- Conditional feature explanation: [SHAP](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)
- Structure learning: [pgmpy](https://pgmpy.org/index.html)

## Models
[conf/models.py](/conf/models.py)

## Configuration
[conf/config.yaml](/conf/config.yaml)

## Execution
`python src/entrypoint.py`
