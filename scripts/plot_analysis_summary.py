"""Recreate figures in analysis.md from existing aggregate artifacts."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.manifold import TSNE
from utils.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = ROOT / 'img/analysis'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
                     'mathtext.fontset': 'stix', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130})


def load(path):
    return json.loads((ROOT / 'artifacts' / path).read_text())


def save(name):
    plt.savefig(OUT / name, bbox_inches='tight', dpi=160)
    plt.close()


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    mid = (p + z*z / (2*n)) / (1 + z*z/n)
    err = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return p-mid+err, mid+err-p


def personas():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, source in zip(axes, ['electronics', 'cosmetics']):
        data = load(f'two-dataset-gru-clusters/{source}.json')
        groups = next(r['groups'] for r in data['runs'] if r['method'] == 'kmeans' and r['k'] == 6)
        x = np.arange(len(groups))
        for offset, action, color in [(-.18, 'cart', '#3d6b91'), (.18, 'purchase', '#d58b39')]:
            values = [100*g['users_with_action'][action] for g in groups]
            ax.bar(x+offset, values, width=.34, label=action, color=color)
        ax.set_xticks(x, [f"c{g['cluster']}\nn={g['users']}" for g in groups])
        ax.set_title(source.title())
        ax.set_ylim(0, 110)
        ax.set_xlabel('Dataset-specific cluster / held-out users')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Users with action in input prefix (%)')
    axes[1].legend(frameon=False)
    fig.suptitle('K-means K=6: observed prefix actions, not future preference')
    fig.tight_layout()
    save('persona_prefix.png')


def evolution():
    data = load('persona-evolution/electronics.json')
    matrix = np.asarray(data['early_to_later_counts'])
    assert matrix.sum() == data['users']
    fig, ax = plt.subplots(figsize=(6, 4.5))
    image = ax.imshow(matrix, cmap='Blues', vmin=0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > matrix.max()*.6 else 'black')
    ax.set_xticks(range(3), ['c0', 'c1', 'c2'])
    ax.set_yticks(range(3), ['c0', 'c1', 'c2'])
    ax.set(xlabel='Sessions 3–4 label', ylabel='Sessions 1–2 label',
           title='Electronics: window labels (n=351; K=3)')
    fig.colorbar(image, ax=ax, label='Users')
    fig.tight_layout()
    save('early_late_labels.png')


def inactivity():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, source in zip(axes, ['electronics', 'cosmetics']):
        rows = load(f'late-inactivity/{source}.json')['profiles']
        rows = sorted(rows, key=lambda x: x['inactive_30d'])
        x = np.arange(3)
        for idx, row in enumerate(rows):
            n = row['users']
            counts = [row['third_session_types'].get('view_only', 0), row['third_cart_any'], row['third_purchase_any']]
            vals = np.array(counts)/n
            errs = np.array([wilson(k, n) for k in counts]).T
            ax.bar(x + (idx-.5)*.36, 100*vals, .36, yerr=100*errs, capsize=3,
                   color=['#3d6b91', '#d58b39'][idx], label=('Reobserved' if idx == 0 else 'No event 30d')+f' (n={n:,})')
        ax.set_xticks(x, ['View only', 'Any cart', 'Any purchase'])
        ax.set_title(source.title())
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('Users in group (%)')
    fig.suptitle('Third observed session; error bars: Wilson 95% intervals')
    fig.tight_layout()
    save('inactivity_profiles.png')


def coupon_return():
    rows = load('coupon-reactivation/summary.json')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, group in enumerate(['click_only', 'click_purchase', 'purchase_only']):
        for received in [0, 1]:
            r = next(x for x in rows if x['group'] == group and x['anchor_receipt'] == received)
            p = r['return_7d']/r['n']
            lo, hi = r['return_7d_ci']
            ax.errorbar(i+(received-.5)*.22, 100*p, yerr=[[100*(p-lo)], [100*(hi-p)]],
                        fmt='o' if received else 's', color='#d58b39' if received else '#3d6b91', capsize=4)
            ax.annotate(f"n={r['n']:,}", (i+(received-.5)*.22, 100*p), xytext=(5, 6),
                        textcoords='offset points', fontsize=8)
    ax.plot([], [], 's', color='#3d6b91', label='No recorded receipt')
    ax.plot([], [], 'o', color='#d58b39', label='Recorded receipt')
    ax.set_xticks(range(3), ['Prior click only', 'Prior click + purchase', 'Prior purchase only'])
    ax.set(ylabel='Recorded activity within 7 days (%)', ylim=(0, 45),
           title='Tianchi: 14-day quiet cohort, observational comparison')
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save('coupon_reactivation.png')


def coupon_weeks():
    rows = load('uplift-sequence-pilot/long_term_observed.json')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, group in zip(axes, ['click_only', 'click_purchase', 'purchase_only']):
        for receipt in [0, 1]:
            row = next(r for r in rows if r['group'] == group and r['receipt_on_anchor'] == receipt)
            ax.plot([w['week'] for w in row['weeks']], [100*w['purchase_any'] for w in row['weeks']],
                    marker='o' if receipt else 's', color='#d58b39' if receipt else '#3d6b91',
                    label=('Receipt' if receipt else 'No receipt') + f" (n={row['n']:,})")
        ax.set_title(group.replace('_', ' ').title())
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel('Week after anchor')
        ax.set_ylim(0, 35)
        ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('Users with recorded purchase (%)')
    fig.suptitle('Tianchi: weekly purchase records; not coupon effects')
    fig.tight_layout()
    save('coupon_weekly_purchase.png')


def agent_benchmark():
    data = load('agent-benchmark/summary.json')
    if 'benchmarks' in data:
        return multi_domain_agent_benchmark(data)
    agent_names = ['UCB', 'TS', 'LinUCB', 'LinTS', 'PPO', 'SAC']
    baseline_names = ['Never coupon', 'Max allowed']
    colors = {
        'UCB': '#54789b', 'TS': '#7896b2', 'LinUCB': '#c99a43',
        'LinTS': '#e0b765', 'PPO': '#a66a87', 'SAC': '#c65d3e',
        'Never coupon': '#b7bcc2', 'Max allowed': '#5f666d',
    }

    labels = agent_names + baseline_names
    test_rows = [data['agents'][name]['test'] for name in agent_names]
    test_rows += [data['test_baselines'][name] for name in baseline_names]
    profit = np.array([row['mean_net_profit'] for row in test_rows])
    error = np.array([row['standard_error_net_profit'] for row in test_rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(labels))
    axes[0].bar(x, profit, yerr=error, capsize=3,
                color=[colors[name] for name in labels])
    axes[0].set_xticks(x, labels, rotation=35, ha='right')
    axes[0].set_ylabel('Annual net profit per user')
    axes[0].set_title('Frozen-policy evaluation (5,000 users)')
    axes[0].set_ylim(0, max(profit + error) * 1.12)
    axes[0].grid(axis='y', alpha=.2)
    for idx, value in enumerate(profit):
        axes[0].text(idx, value + error[idx] + 550, f'{value:,.0f}',
                     ha='center', va='bottom', fontsize=8, rotation=90)

    positions = {}
    for name, row in zip(labels, test_rows):
        axes[1].scatter(row['mean_coupons'], row['mean_purchases'], s=70,
                        color=colors[name], edgecolor='white', linewidth=.7, zorder=3)
        position = (row['mean_coupons'], row['mean_purchases'])
        positions.setdefault(position, []).append(name)
    for position, names in positions.items():
        label = ' / '.join(names)
        single_offsets = {'SAC': (6, 8), 'LinTS': (6, 0), 'LinUCB': (6, -9)}
        offset = (-105, 8) if len(names) > 1 else single_offsets.get(names[0], (6, 5))
        axes[1].annotate(label, position, xytext=offset,
                         textcoords='offset points', fontsize=8)
    axes[1].set(xlabel='Coupons per user / year', ylabel='Purchases per user / year',
                title='Coupon use and realized purchases')
    axes[1].set_xlim(-2, 56)
    axes[1].set_ylim(0, max(row['mean_purchases'] for row in test_rows) * 1.14)
    axes[1].grid(alpha=.2)
    fig.suptitle('Weekly coupon policy benchmark; error bars are standard errors')
    fig.tight_layout()
    save('agent_policy_comparison.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for name in agent_names:
        rows = data['agents'][name]['online_learning_curve']
        users = [row['users'] for row in rows]
        axes[0].plot(users, [row['cumulative_net_profit'] / 1e6 for row in rows],
                     marker='o', color=colors[name], label=name)
        axes[1].plot(users, [row['learning_cost_vs_canonical_oracle'] / 1e6 for row in rows],
                     marker='o', color=colors[name], label=name)
    axes[0].set(title='Cumulative training net profit', ylabel='Million profit units')
    axes[1].set(title='Learning cost vs canonical oracle', ylabel='Million profit units')
    for ax in axes:
        ax.set_xlabel('Training users processed')
        ax.set_xticks([100, 500, 1000])
        ax.grid(alpha=.2)
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle('Online learning over 1,000 users × 13 weeks')
    fig.tight_layout()
    save('agent_learning_curve.png')


def multi_domain_agent_benchmark(data):
    """Plot the three-domain, two-cost benchmark from the final protocol."""
    agents = ['UCB', 'TS', 'LinUCB', 'LinTS', 'PPO', 'SAC']
    domains = ['cosmetics', 'electronics', 'tianchi']
    costs = [('send_cost', 'Send cost'),
             ('redemption_discount', 'First-purchase discount')]
    colors = ['#54789b', '#7896b2', '#c99a43', '#e0b765', '#a66a87', '#c65d3e']
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True)
    for row, (cost, cost_label) in enumerate(costs):
        for column, domain in enumerate(domains):
            result = data['benchmarks'][f'{domain}:{cost}']
            summaries = [result['agents'][agent]['test'] for agent in agents]
            means = np.array([entry['mean_net_profit'] for entry in summaries]) / 1000
            errors = np.array([
                entry['standard_error_net_profit'] for entry in summaries
            ]) / 1000
            axis = axes[row, column]
            axis.bar(np.arange(len(agents)), means, yerr=errors, capsize=2,
                     color=colors, width=.72)
            axis.set_title(f'{domain.title()} — {cost_label}')
            axis.grid(axis='y', alpha=.2)
            if column == 0:
                axis.set_ylabel('Net profit per user (×1,000)')
            axis.set_xticks(np.arange(len(agents)), agents, rotation=35, ha='right')
    fig.suptitle('Frozen-policy evaluation: 5,000 users × 52 weeks')
    fig.tight_layout()
    save('agent_policy_comparison.png')

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    for axis, (cost, cost_label) in zip(axes, costs, strict=True):
        x = np.arange(len(agents))
        width = .25
        for index, domain in enumerate(domains):
            result = data['benchmarks'][f'{domain}:{cost}']
            values = [
                result['agents'][agent]['test']['mean_coupon_cost_per_purchase']
                for agent in agents
            ]
            axis.bar(x + (index - 1) * width, values, width, label=domain.title())
        axis.set_title(cost_label)
        axis.set_ylabel('Coupon cost per realized purchase')
        axis.set_xticks(x, agents, rotation=35, ha='right')
        axis.grid(axis='y', alpha=.2)
    axes[1].legend(frameon=False)
    fig.suptitle('Coupon economics under the frozen policies')
    fig.tight_layout()
    save('agent_coupon_cost_per_purchase.png')


def operational_persona_event_mix():
    data = load('cosmetics-shop-personas/clusters.json')
    groups = sorted(data['profiles'], key=lambda row: row['cluster'])
    actions = ['view', 'cart', 'remove', 'purchase']
    colors = ['#4C78A8', '#F58518', '#54A24B', '#B279A2']
    x = np.arange(len(actions), dtype=float)
    dense_x = np.linspace(x.min(), x.max(), 240)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for row, color in zip(groups, colors, strict=True):
        values = np.array([100 * row['event_share'][action] for action in actions])
        smooth = PchipInterpolator(x, values)(dense_x)
        label = f"P{row['cluster']} (n={row['users']:,})"
        ax.plot(dense_x, smooth, color=color, linewidth=2.2, label=label)
        ax.scatter(x, values, color=color, edgecolor='white', linewidth=.7,
                   s=34, zorder=3)
    ax.set_xticks(x, actions)
    ax.set(xlabel='Event type', ylabel='Within-cluster event frequency (%)',
           ylim=(0, 100), title='Cosmetics K=4 clusters: event-frequency profiles')
    ax.grid(axis='y', alpha=.22)
    ax.legend(frameon=False, ncol=2, loc='upper right')
    fig.tight_layout()
    save('cosmetics_k4_event_frequency.png')


def canonical_type_fit():
    model = load('tianchi-uplift-model/model.json')
    rows = model['canonical_assignment']['persona_fit']
    types = model['canonical_assignment']['types']
    labels = ['Sure thing', 'Persuadable', 'Sleeping dog', 'Lost cause']
    colors = {0: '#4C78A8', 2: '#54A24B', 3: '#B279A2'}
    x = np.arange(len(types), dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    width = .24
    for index, row in enumerate(rows):
        persona = int(row['persona'])
        target = 100 * np.array([row['target'][name] for name in types])
        color = colors[persona]
        ax.bar(x + (index - 1) * width, target, width=width,
               color=color, label=f'P{persona}')
    ax.set_xticks(x, ['ST', 'P', 'SD', 'LC'])
    ax.set(xlabel='Canonical type', ylabel='Within-persona probability (%)',
           ylim=(0, 82), title='Canonical-type distributions by persona')
    ax.grid(axis='y', alpha=.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    save('canonical_type_latent_fit.png')


def simulator_latent_tsne():
    source = ROOT / 'artifacts/tianchi-uplift-model/sampled_user_uplift_profiles.parquet'
    cache = ROOT / 'artifacts/tianchi-uplift-model/simulator_latent_tsne.parquet'
    if cache.exists():
        frame = pd.read_parquet(cache)
    else:
        profiles = pd.read_parquet(source)
        z_columns = sorted(
            (column for column in profiles if column.startswith('z_')),
            key=lambda column: int(column[2:]),
        )
        coordinates = TSNE(
            n_components=2, perplexity=50, learning_rate='auto', init='pca',
            max_iter=1000, random_state=42,
        ).fit_transform(profiles[z_columns].to_numpy(float))
        frame = profiles[['user_id', 'home_persona']].copy()
        frame[['tsne_1', 'tsne_2']] = coordinates
        frame.to_parquet(cache, index=False)
    colors = {0: '#4C78A8', 2: '#54A24B', 3: '#B279A2'}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for persona, group in frame.groupby('home_persona', sort=True):
        ax.scatter(group.tsne_1, group.tsne_2, s=8, alpha=.35,
                   color=colors[int(persona)], edgecolors='none', rasterized=True,
                   label=f'P{int(persona)} (n={len(group):,})')
    ax.set(xlabel='t-SNE 1', ylabel='t-SNE 2',
           title='Simulated user latents by home persona')
    ax.legend(frameon=False, markerscale=2.2)
    ax.grid(alpha=.15)
    fig.tight_layout()
    save('simulator_latent_tsne.png')


def simulator_distribution_figure():
    model = load('tianchi-uplift-model/model.json')
    rows = model['canonical_assignment']['persona_fit']
    types = model['canonical_assignment']['types']
    frame = pd.read_parquet(
        ROOT / 'artifacts/tianchi-uplift-model/simulator_latent_tsne.parquet'
    )
    colors = {0: '#4C78A8', 2: '#54A24B', 3: '#B279A2'}
    x = np.arange(len(types), dtype=float)
    dense_x = np.linspace(x.min(), x.max(), 240)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    for row in rows:
        persona = int(row['persona'])
        values = 100 * np.array([row['target'][name] for name in types])
        axes[0].plot(dense_x, PchipInterpolator(x, values)(dense_x),
                     color=colors[persona], linewidth=2.2, label=f'P{persona}')
        axes[0].scatter(x, values, color=colors[persona], edgecolor='white',
                        linewidth=.7, s=34, zorder=3)
    axes[0].set_xticks(x, ['ST', 'P', 'SD', 'LC'])
    axes[0].set(xlabel='Canonical type', ylabel='Within-persona probability (%)',
                ylim=(0, 82), title='(a) Canonical-type distributions')
    axes[0].grid(axis='y', alpha=.22)
    axes[0].legend(frameon=False)

    for persona, group in frame.groupby('home_persona', sort=True):
        axes[1].scatter(group.tsne_1, group.tsne_2, s=8, alpha=.35,
                        color=colors[int(persona)], edgecolors='none', rasterized=True,
                        label=f'P{int(persona)} (n={len(group):,})')
    axes[1].set(xlabel='t-SNE 1', ylabel='t-SNE 2',
                title='(b) Simulated user latents')
    axes[1].legend(frameon=False, markerscale=2.2)
    axes[1].grid(alpha=.15)
    save('figure3_simulated_user_distribution.png')


def domain_transfer_and_canonical():
    transfer = pd.read_parquet(
        ROOT / 'artifacts/domain-simulator-profiles/user_persona_transfer.parquet'
    )
    proxy = pd.read_csv(
        ROOT / 'artifacts/tianchi-canonical-proxies/canonical_type_proxies.csv'
    )
    model = load('domain-simulator-profiles/model.json')
    colors = {0: '#4C78A8', 2: '#E3A13B', 3: '#6BA56B'}
    canonical_colors = ['#4C78A8', '#E3A13B', '#B279A2', '#A7A7A7']
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), constrained_layout=True)

    domains = ['cosmetics', 'electronics', 'tianchi']
    bottom = np.zeros(len(domains))
    for persona in (0, 2, 3):
        values = np.array([
            100 * transfer.loc[transfer.dataset.eq(domain), 'transferred_persona']
            .eq(persona).mean() for domain in domains
        ])
        axes[0].bar(domains, values, bottom=bottom, color=colors[persona],
                    label=f'P{persona}', width=.62)
        bottom += values
    components = [len(model['domains'][domain]['gmm']['weights']) for domain in domains]
    for index, count in enumerate(components):
        axes[0].text(index, 102, f'{count}-component GMM', ha='center', va='bottom', fontsize=8)
    axes[0].set(ylabel='Transferred users (%)', ylim=(0, 112),
                title='(a) Domain sampling mixture')
    axes[0].legend(frameon=False, ncol=3, loc='lower center')
    axes[0].grid(axis='y', alpha=.2)

    groups = ['P0', 'P2', 'P3', 'P_LT']
    types = ['sure_thing', 'persuadable', 'sleeping_dog', 'lost_cause']
    labels = ['ST', 'P', 'SD', 'LC']
    x = np.arange(len(groups))
    width = .19
    for index, (kind, label, color) in enumerate(zip(types, labels, canonical_colors)):
        values = [
            100 * float(proxy.loc[proxy.persona.eq(group), kind].iloc[0])
            for group in groups
        ]
        axes[1].bar(x + (index - 1.5) * width, values, width,
                    color=color, label=label)
    axes[1].set_xticks(x, groups)
    axes[1].set(ylabel='Canonical-type probability (%)', ylim=(0, 78),
                title='(b) Matched Tianchi calibration')
    axes[1].legend(frameon=False, ncol=4, loc='upper center')
    axes[1].grid(axis='y', alpha=.2)
    save('domain_transfer_and_canonical.png')


if __name__ == '__main__':
    personas()
    evolution()
    inactivity()
    coupon_return()
    coupon_weeks()
    agent_benchmark()
    operational_persona_event_mix()
    canonical_type_fit()
    simulator_latent_tsne()
    simulator_distribution_figure()
    domain_transfer_and_canonical()
