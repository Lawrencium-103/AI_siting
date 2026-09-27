# ai-siting: analysis code for "Siting large AI training runs"

This repository contains the complete analysis behind the paper:

> Siting large AI training runs: a calibrated, hour-resolved assessment of how grid, climate and cooling shape carbon and water footprints across six global locations (in review).

One command downloads the public input data, runs every model and writes every table, figure and number reported in the paper. Nothing is typed by hand.

The reusable models are also released separately as **SiteFootprint**, a Python package and web application ([doi:10.5281/zenodo.22983788](https://doi.org/10.5281/zenodo.22983788), `pip install sitefootprint`). This repository is the frozen research record; SiteFootprint is the tool for applying the method to new runs and locations.

## Reproduce the paper

    pip install -r requirements.txt
    python run_all.py --fallback-cohort data/master_gold.parquet

The first run downloads and caches, under `data/raw/`:

- hourly weather for 2021–2025 at 20 locations (Open-Meteo historical archive, ERA5-based)
- Ember annual grid intensity (via the Our World in Data energy dataset on GitHub)
- the Epoch AI list of large-scale training runs (falls back to `data/master_gold.parquet` if unreachable)

Later runs work offline. A full run takes about 5 minutes on a laptop (1,000 Monte Carlo draws, 256 Sobol base samples).

Tests of the physical model, with no network needed:

    pytest -q

Test the code path without internet, using synthetic weather (all outputs are watermarked DEMO; never report them):

    python run_all.py --demo --offline --fallback-cohort data/master_gold.parquet --n-mc 200 --n-sobol 64

## What each output corresponds to in the paper

| Output | Paper |
|---|---|
| `outputs/tables/table1_sites.csv` | Table 1: climate and grid characteristics of the sites |
| `outputs/tables/table2_parameters.csv` | Table 2: parameter distributions and sources |
| `outputs/tables/table3_validation.csv` | Table 3: Llama 3.1 405B validation and PUE calibration |
| `outputs/tables/table4_site_results.csv` | Table 4: PUE, water and emissions by site and cooling design |
| `outputs/tables/table5_decomposition.csv` | Table 5: grid-versus-facility decomposition, paired probabilities |
| `outputs/tables/table6_siting.csv` | Table 6: capacity-constrained siting of the cohort |
| `outputs/tables/tableS1` … `tableS9` | Supplementary Tables S1–S9 |
| `outputs/figures/fig1_climate.png` … `fig6_diesel_siting.png` | Figures 1–6 |
| `outputs/results.json` | every number quoted in the text |

The `outputs/` folder in this archive holds the results reported in the submitted manuscript.

## Code

| File | Role |
|---|---|
| `ai_siting/config.py` | sites, years, data sources, and every parameter with its source |
| `ai_siting/data.py` | downloading and caching of weather, grid intensity and the training-run cohort |
| `ai_siting/physics.py` | wet-bulb temperature, the hourly cooling model (three architectures), IT energy, carbon |
| `ai_siting/sampling.py` | Latin hypercube Monte Carlo and Sobol designs |
| `ai_siting/analysis.py` | calibration and hold-out test, Llama validation, Monte Carlo, Sobol, decomposition, diesel scenarios |
| `ai_siting/siting.py` | capacity-constrained siting (MILP, HiGHS via SciPy) and the carbon–water Pareto front |
| `ai_siting/figures.py` | all figures |
| `ai_siting/derived.py` | derived reporting quantities computed from the output tables |
| `run_all.py` | runs everything in order |
| `data/google_campus_pue.csv` | calibration targets (Google trailing-twelve-month PUE, 2021–2025) with source URLs |
| `data/master_gold.parquet` | the authors' earlier Epoch AI extract, used only if the live Epoch CSV is unavailable |

## Data sources and licences

| Data | Source | Licence |
|---|---|---|
| Hourly weather | Open-Meteo historical archive (ERA5 reanalysis) | CC BY 4.0 |
| Grid carbon intensity | Ember yearly electricity data, via Our World in Data | CC BY 4.0 |
| Training runs | Epoch AI, Data on Large-Scale AI Models | CC BY 4.0 |
| Campus PUE | Google data-centre efficiency disclosures (https://datacenters.google/efficiency) | public disclosure |

## Changes since the submitted results

- **1.0.0:** In the Llama 3.1 405B validation, the Ashburn PUE is now computed for every Monte Carlo draw. Previously the first 300 draws were repeated, which paired some PUE values with the wrong IT-energy parameters. The effect on the validation emissions interval is below 1%; no other number changes.

## Citation

See `CITATION.cff`. Please also cite the paper once published, and SiteFootprint if you use the package.

## Licence

MIT (code). Input data remain under their original licences listed above.
