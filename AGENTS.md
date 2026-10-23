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

**Project Tree(Simplified)**
```md
.
├── AGENTS.md
├── README.md
├── analysis.md
├── .gitignore
├── requirements.txt
├── data
│   ├── raw
│   │   ├── electronics
│   │   ├── cosmetics
│   │   └── tianchi
│   ├── augmented
│   │   ├── persona
│   │   ├── uplift
│   │   └── delay
├── src
│   ├── analysis
│   ├── agent (This is needed as we need to run various types of bandits and Deep RL agents)
│   ├── simulate
│   └── utils
└── img
```

As you can see, current repository is messed up. After refactoring, and summarize important analysis, and result of each goal into analysis.md., delete unnecessary python files for testing, and analysis (This must be cleaned up)