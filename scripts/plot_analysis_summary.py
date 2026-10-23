"""Recreate figures in analysis.md from existing aggregate artifacts."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ecommerce_sim.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = ROOT / 'research/figures/analysis'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 130})


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


if __name__ == '__main__':
    personas()
    evolution()
    inactivity()
    coupon_return()
    coupon_weeks()
