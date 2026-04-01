### Epidemic Progression Modelling: CodeCure

--- TODO: @Ujan ---

### Getting Started

```pip install -r requirements.txt```

```python preprocessing.py```

```python model.py```

Modify ```COUNTRY_ISO2 = "BD" ``` and ```COUNTRY_ISO3 = "BGD" ``` with ISO-2 and ISO-3 country codes at ```main.py``` to access population densities of different countries.

--- TODO (@Ujan): Add more detailed docs ---

### Conventions

Always update ```/requirements.txt``` and ```/dockerfile``` and build the app before sending PRs.

TODO (@Ujan) : Add naming and contributions conventions for ```/helpers```

**File naming conventions -**

- In the folder ```/helpers``` all the helpers related to data downloading has the name ```data_XXX.py```

- In the folder ```/visual_analysis``` all the helpers related to data downloading has the name ```analysis_XXX.py```

### Data Notes

- ```helpers/data_covid.py``` downloads and caches Google COVID-19 Open Data epidemiology and geography datasets.

- ```helpers/data_mobility.py``` downloads and caches Google COVID-19 Open Data mobility datasets.

- Country-specific location lookups are cached under ```data/covid/location_lookup/``` so location keys are processed once and reused for faster joins.
