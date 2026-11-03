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

**Project Tree (Current, Simplified)**
```text
.
├── AGENTS.md
├── README.md
├── analysis.md
├── pyproject.toml
├── data/raw/                   # local datasets
├── artifacts/                  # local experiment outputs
├── src/
│   ├── analysis/
│   │   ├── irl/
│   │   │   └── history.py      # history-based IRL experiments
│   │   ├── persona/
│   │   │   ├── sequence.py     # sequence representation and masked GRU
│   │   │   ├── cluster.py      # clustering, K-sweep, stability
│   │   │   └── behavior.py     # behavior, session gaps, heterogeneity
│   │   ├── uplift/
│   │   │   ├── canonical.py    # assumption-dependent response bounds
│   │   │   ├── observed.py     # descriptive coupon analyses
│   │   │   └── sequence.py     # observational sequence prediction
│   │   ├── eda.py              # data audits, inactivity, session returns, causal features
│   │   └── feature_selection.py # past-only feature selection and session structure exploration
│   ├── preprocessing/
│   ├── models/
│   │   └── agents.py           # UCB, TS, LinUCB, LinTS, PPO, discrete SAC
│   ├── simulate/
│   │   ├── latent_session_model.py
│   │   └── campaign.py         # daily capped coupon-policy environment
│   ├── utils/
│   └── cli.py
├── scripts/
├── tests/
├── img/
├── external/
├── paper/
└── output/
```

As you can see, current repository is messed up. After refactoring, and summarize important analysis, and result of each goal into analysis.md., delete unnecessary python files for testing, and analysis (This must be cleaned up)

**Working Rules**

- Organize code by domain and reusable function, not by dataset or one-off experiment. Keep persona sequence, clustering, and behavior code together under `src/analysis/persona/`; avoid unnecessary nested project-name packages.
- Keep scripts as necessary entry points. Consolidate redundant analysis scripts and reusable utilities, and update callers, tests, and documentation when refactoring.
- Keep analysis results, assumptions, and limitations in `analysis.md`; place figures in `img/` rather than creating a separate research documentation tree.
- Distinguish observed patterns, simulation assumptions, and identified causal effects. Separate interventions from user actions, prevent future-information leakage, and do not treat missing observations as zero outcomes or confirmed inactivity.
- Persona clusters do not identify canonical response types or individual counterfactuals. State the assumptions and limitations of any uplift or IRL-based model explicitly.
- Preserve user data and useful reproducibility; verify changes and report what was not tested. Check dataset usage and redistribution rights before publication.
