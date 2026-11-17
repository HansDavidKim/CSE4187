### Data Preprocessing & EDA
**Goal** 
1) Using Electronics & Cosmetics, we extract different type of Persona using clustering.
2) Using Tianchi as supplement, we generate uplift hypothesis : CATE analysis
3) CATE analysis includes canonical type distribution analysis : sure-thing, persuadable, sleeping dog, lost cause.
4) Also, period between sessions of the identical user analysis is contained.
5) When goal 1 to 4 are finished, we then do feature engineering : which variables should be used for prediction
6) After that, we make hypothesis of utility function for each behavior including session end.
7) After the session is done, the user model samples delay between two session.
8) Inverse reinforcement learning with maximum entropy method is adopted due to heterogenous persona groups.
9) Estimate each group's parameters : e.g. Q(s, a) = sum of linear combinations of features where coefficient of each feature is parameter.
10) With Tianchi dataset, you must augment uplift term (it can be vector or etc). Also, this augmented formula must be able to replicate the reulst without uplift, and also it must be able to replicate the user behavior with uplift.
11) Period-dealy heterogeneity must be dealt with, and we assume latent factor s.t. its value is updated after the each session. This value determines the gap between the session (and its delay generating process must be stochastic).
12) After estimation is done, you must estimate each type's distribution, and design sampler mechanism as Gaussian Mixture Model.
13) Both offline dataset generator and online simulator must be available.
14) Determine features that are visible to each agent.
15) Determine goal or objective for the task : Reference other simulator papers, and determine one.
16) Make scripts for running various RL agents : Lin-UCB, Lin-TS, TS, UCB, PPO, SAC.

**Tracked Project Tree**
```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── src/
│   ├── analysis/             # existing research functions
│   ├── models/               # existing learning and policy code
│   ├── preprocessing/
│   ├── simulate/
│   │   └── v4/               # configuration, simulator, IRL, data, features and validation
│   ├── utils/
│   └── cli.py
├── scripts/                  # necessary entry points and SQL examples
└── tests/
```

Raw data, models, generated outputs, reports, notebooks, environments and caches
stay local and are excluded from Git. Preserve these local files when changing
repository tracking. Keep required runtime configuration under `src/`.

**Working Rules**

- Organize code by domain and reusable function, not by dataset or one-off experiment. Keep persona sequence, clustering, and behavior code together under `src/analysis/persona/`; avoid unnecessary nested project-name packages.
- Keep scripts as necessary entry points. Consolidate redundant analysis scripts and reusable utilities, and update callers, tests, and documentation when refactoring.
- Keep experiment results, reports and figures in ignored local directories. Keep concise runtime instructions in `README.md`; do not commit generated research documentation.
- Distinguish observed patterns, simulation assumptions, and identified causal effects. Separate interventions from user actions, prevent future-information leakage, and do not treat missing observations as zero outcomes or confirmed inactivity.
- Persona clusters do not identify canonical response types or individual counterfactuals. State the assumptions and limitations of any uplift or IRL-based model explicitly.
- Preserve user data and useful reproducibility; verify changes and report what was not tested. Check dataset usage and redistribution rights before publication.
