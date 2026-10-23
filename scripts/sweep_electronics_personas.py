"""Profile held-out Electronics prefixes for multiple cluster counts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

if __package__:
    from scripts.run_electronics_personas import OUT, SOURCE
else:
    from run_electronics_personas import OUT, SOURCE


def main():
    from src.irl.fit import load_events

    from src.irl.sequence_personas import sequence_rows

    embeddings = pd.read_parquet(OUT / 'electronics_embeddings.parquet')
    events, _ = load_events('electronics', SOURCE)
    rows = sequence_rows(events, max_users=12000, max_tokens=64, seed=42)
    if embeddings.user_id.tolist() != [row[0] for row in rows]:
        raise ValueError('Embeddings and reconstructed sample order do not match')
    z = embeddings[[f'z_{i}' for i in range(48)]].to_numpy()
    sessions = np.array([row[3] for row in rows])
    train, test = train_test_split(np.arange(len(rows)), test_size=0.2, random_state=42)
    split = np.full(len(rows), 'train', dtype=object)
    split[test] = 'test'
    memberships = pd.DataFrame({'user_id': embeddings.user_id, 'split': split,
                                'full_history_sessions': sessions})
    train = train[sessions[train] >= 2]
    test = test[sessions[test] >= 2]
    profiles = pd.DataFrame([
        {'views': rows[i][1].count(0), 'carts': rows[i][1].count(1),
         'purchases': rows[i][1].count(2), 'tokens': len(rows[i][1]),
         'view_next': sum(a == 0 for a, b in zip(rows[i][1], rows[i][1][1:])),
         'view_to_cart': sum(a == 0 and b == 1 for a, b in zip(rows[i][1], rows[i][1][1:])),
         'cart_next': sum(a == 1 for a, b in zip(rows[i][1], rows[i][1][1:])),
         'cart_to_purchase': sum(a == 1 and b == 2 for a, b in zip(rows[i][1], rows[i][1][1:]))}
        for i in test
    ])
    sweep = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(z[train])
        memberships[f'k{k}'] = model.predict(z)
        labels = model.predict(z[test])
        alternative = KMeans(n_clusters=k, n_init=10, random_state=43).fit(z[train])
        groups = []
        for cluster in range(k):
            group = profiles[labels == cluster]
            counts = group[['views', 'carts', 'purchases']].sum()
            total = int(counts.sum())
            groups.append({
                'cluster': cluster, 'heldout_users': len(group),
                'event_share': (counts / total).to_dict() if total else None,
                'buyer_share': float(group.purchases.gt(0).mean()) if len(group) else None,
                'cart_user_share': float(group.carts.gt(0).mean()) if len(group) else None,
                'median_tokens': float(group.tokens.median()) if len(group) else None,
                'view_to_cart_next_rate': (float(group.view_to_cart.sum() / group.view_next.sum())
                                           if group.view_next.sum() else None),
                'cart_to_purchase_next_rate': (float(group.cart_to_purchase.sum() / group.cart_next.sum())
                                               if group.cart_next.sum() else None),
                'view_next_count': int(group.view_next.sum()),
                'cart_next_count': int(group.cart_next.sum()),
                'train_users': int((model.labels_ == cluster).sum()),
            })
        assert sum(group['heldout_users'] for group in groups) == len(test)
        sweep.append({'k': k, 'heldout_users': len(test), 'train_users': len(train),
                      'kmeans_seed_42_vs_43_heldout_ari': float(adjusted_rand_score(
                          labels, alternative.predict(z[test]))), 'clusters': groups})
    output = {'source': 'electronics', 'embedding': 'existing seed-42 masked-GRU',
              'scope': 'held-out users with >=2 full-history sessions; first 64 input tokens',
              'limitations': ['stratified discovery sample, not population prevalence',
                              'profiles reuse input, not future outcomes',
                              'ARI changes clustering initialization only, not encoder or sample',
                              'session eligibility uses full observation history'],
              'sweep': sweep}
    (OUT / 'k_sweep.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    memberships.to_parquet(OUT / 'k_sweep_memberships.parquet', index=False)
    for item in sweep:
        print(f"k={item['k']} seed-ARI={item['kmeans_seed_42_vs_43_heldout_ari']:.3f}")
        for group in item['clusters']:
            s = group['event_share']
            if s is None:
                print(f"  c{group['cluster']} train={group['train_users']} test=0")
                continue
            print(f"  c{group['cluster']} train={group['train_users']} test={group['heldout_users']} "
                  f"view/cart/buy={s['views']:.1%}/{s['carts']:.1%}/{s['purchases']:.1%} "
                  f"buyers={group['buyer_share']:.1%} cart-users={group['cart_user_share']:.1%}")


if __name__ == '__main__':
    main()
