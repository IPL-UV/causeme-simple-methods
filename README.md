# CauseMe Simple Methods

A collection of simple causal discovery methods for time series.
Methods implementation is compatible with CauseMe platform.


## Available Methods

- linear VAR
- granger2d

## Template

This repository gives also a fully working template or skeleton
for new [CauseMe](https://causeme.uv.es) python method(s).

```
├── .pre-commit-config.yaml
├── .gitignore
├── causemesplmthds
│   ├── __init__.py
│   ├── granger.py
│   └── var.py
├── dev_requirements.txt
├── LICENSE.txt
├── methods.json
├── pyproject.toml
├── README.md
├── requirements.txt
├── setup.cfg
└── tests
    ├── __init__.py
    └── test_method.py
```

## how to implement new CausMe methods
