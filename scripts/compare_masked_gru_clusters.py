"""Compare clusterers on saved dataset-specific masked-GRU representations."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from threadpoolctl import threadpool_limits

from src.analysis.masked_embedding_clusters import rows_from_events, source_data

ROOT = Path('/Users/kimdaewon/Desktop/eCommerce-Sim')
CAPSTONE = Path('/Users/kimdaewon/Desktop/캡스톤디자인')
OUT = ROOT / 'artifacts/two-dataset-gru-clusters'


def profile(tokens, actions, labels, k):
    records = np.array([[seq.count(i) for i in range(len(actions))] for seq in tokens], dtype=int)
    result = []
    for cluster in range(k):
        group = records[labels == cluster]
        n = len(group)
        total = group.sum()
        pairs = np.zeros((len(actions), len(actions)), dtype=int)
        for seq, label in zip(tokens, labels):
            if label == cluster:
                for a, b in zip(seq, seq[1:]):
                    if a < len(actions) and b < len(actions):
                        pairs[a, b] += 1
        top_pairs = sorted(((int(pairs[i, j]), actions[i], actions[j])
                            for i in range(len(actions)) for j in range(len(actions))), reverse=True)[:3]
        result.append({'cluster': cluster, 'users': n,
                       'event_share': {a: float(group[:, i].sum() / total) if total else None
                                       for i, a in enumerate(actions)},
                       'users_with_action': {a: float((group[:, i] > 0).mean()) if n else None
                                             for i, a in enumerate(actions)},
                       'median_actions': float(np.median(group.sum(axis=1))) if n else None,
                       'within_session_adjacent_pairs': int(pairs.sum()),
                       'top_pairs': [{'from': a, 'to': b, 'count': count,
                                      'share': float(count / pairs.sum())} for count, a, b in top_pairs if count]})
    assert sum(x['users'] for x in result) == len(tokens)
    return result


def run(source):
    path = CAPSTONE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics' else 'data/raw/kaggle/electronics/events.csv')
    events, quality, actions = source_data(source, path, sample_modulus=100)
    rows = rows_from_events(events, 12000, 64, len(actions), 42)
    emb = pd.read_parquet(CAPSTONE / f'data/models/{source}_masked_embedding_embeddings.parquet')
    if emb.user_id.tolist() != [r[0] for r in rows]:
        raise ValueError('saved embedding rows do not match reconstructed tokens')
    z = emb[[f'z_{i}' for i in range(48)]].to_numpy()
    sessions = np.array([r[2] for r in rows])
    rng = np.random.default_rng(42)
    order = rng.permutation(len(rows))
    train, test = order[:int(.8 * len(rows))], order[int(.8 * len(rows)):]
    train, test = train[sessions[train] >= 2], test[sessions[test] >= 2]
    subset = np.random.default_rng(97).choice(train, min(1800, len(train)), replace=False)
    ztrain, ztest = z[subset], z[test]
    tokens = [rows[i][1] for i in test]
    graph = kneighbors_graph(ztrain, n_neighbors=10, include_self=False)
    result = {'source': source, 'source_quality': quality, 'actions': actions,
              'selected_users': len(rows), 'training_repeat_users': len(train),
              'fit_sample_users': len(subset), 'heldout_repeat_users': len(test), 'runs': []}
    with threadpool_limits(limits=1):
        for k in (3, 4, 5, 6):
            for method in ('kmeans', 'gmm_diag', 'knn_graph_ward'):
                stability = None
                if method == 'kmeans':
                    first = KMeans(n_clusters=k, n_init=10, random_state=42).fit(ztrain)
                    alt = KMeans(n_clusters=k, n_init=10, random_state=43).fit(ztrain)
                    labels = first.predict(ztest)
                    stability = adjusted_rand_score(labels, alt.predict(ztest))
                elif method == 'gmm_diag':
                    first = GaussianMixture(n_components=k, covariance_type='diag', reg_covar=1e-3,
                                            n_init=3, max_iter=150, random_state=42).fit(ztrain)
                    alt = GaussianMixture(n_components=k, covariance_type='diag', reg_covar=1e-3,
                                          n_init=3, max_iter=150, random_state=43).fit(ztrain)
                    labels = first.predict(ztest)
                    stability = adjusted_rand_score(labels, alt.predict(ztest))
                else:
                    fitted = AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=graph).fit_predict(ztrain)
                    labels = KNeighborsClassifier(n_neighbors=1).fit(ztrain, fitted).predict(ztest)
                result['runs'].append({'method': method, 'k': k,
                                       'holdout_seed_ari': float(stability) if stability is not None else None,
                                       'groups': profile(tokens, actions, labels, k)})
                print(source, method, k, [g['users'] for g in result['runs'][-1]['groups']], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{source}.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')


def report():
    docs = ['# Electronics·Cosmetics GRU 임베딩 기반 사용자 행동 군집', '',
            '## 목적과 방법', '',
            '기존 양방향 GRU masked-token 인코더의 저장된 48차원 임베딩을 데이터셋별로 재사용했다. '
            'BERT 자체가 아니라 양방향 GRU에 masked-token 복원 목적을 적용한 모델이다. '
            '입력은 행동 토큰과 세션 경계 토큰이며 앞 최대 64개 토큰이다. 모델 학습과 임베딩은 새로 수행하지 않았다.', '',
            '기존 사용자 표본 및 80/20 사용자 분할을 재구성하고, 전체 이력 기준 세션 2개 이상인 표본에서 '
            '학습 1,800명 이하로 군집을 학습해 보류 사용자에게 할당했다. '
            'K-means, 대각 공분산 GMM, 10-NN 연결 그래프 Ward 계층 군집을 K=3,4,5,6으로 비교한다. '
            'KNN 자체는 비지도 군집이 아니며 Ward의 보류 사용자 할당에는 1-NN을 사용한다. '
            'ARI는 동일 사용자에서 두 초기화의 할당 일치도이며 encoder·표본 변동 안정성은 아니다.', '',
            '각 비율은 보류 표본의 입력 prefix에서 측정한 기술 통계다. 행동 비율은 이벤트 가중이고, '
            '행동 경험 비율은 사용자 단위로 서로 중복 가능하다. 인접 행동 비율은 해당 군집의 세션 내 인접 행동쌍 전체가 분모다. '
            '이 prefix 자체가 군집에도 사용되므로 독립적 미래 검증이 아니다.', '']
    for source in ('electronics', 'cosmetics'):
        d = json.loads((OUT / f'{source}.json').read_text())
        docs += [f'## {source.title()}', '',
                 f"표본 {d['selected_users']:,}명 중 반복 세션 조건 충족 보류 사용자 {d['heldout_repeat_users']:,}명; 군집 학습 {d['fit_sample_users']:,}명.", '']
        reference = next(r for r in d['runs'] if r['method'] == 'kmeans' and r['k'] == 4)
        substantial = [g for g in reference['groups'] if g['users'] >= 30]
        most_view = max(substantial, key=lambda g: g['event_share']['view'])
        most_buy = max(substantial, key=lambda g: g['users_with_action']['purchase'])
        docs += [f"K-means K=4의 기술적 예시: 군집 {most_view['cluster']}은 조회가 이벤트의 "
                 f"{100*most_view['event_share']['view']:.1f}%인 조회 중심 군집이며, "
                 f"군집 {most_buy['cluster']}은 관측 prefix 구매 경험 비율이 "
                 f"{100*most_buy['users_with_action']['purchase']:.1f}%로 30명 이상 군집 중 가장 높다. "
                 '이 기준은 패턴 설명을 위한 예시이지 K=4가 최적이라는 주장은 아니다.', '']
        if source == 'cosmetics':
            most_remove = max(substantial, key=lambda g: g['users_with_action']['remove'])
            docs += [f"장바구니 삭제 경험 비율은 군집 {most_remove['cluster']}에서 "
                     f"{100*most_remove['users_with_action']['remove']:.1f}%로 높았다. "
                     '삭제 원인이나 불만족 여부는 관측하지 않았다.', '']
        for run in d['runs']:
            ari = '해당 없음' if run['holdout_seed_ari'] is None else f"{run['holdout_seed_ari']:.3f}"
            docs += [f"### {run['method']}, K={run['k']} (초기화 ARI: {ari})", '',
                     '| 군집 | 사용자 수 | 중앙 행동 수 | 조회 비율 | 장바구니 경험 | 구매 경험 | 삭제 경험 | 주요 세션 내 인접 행동 |',
                     '|---|---:|---:|---:|---:|---:|---:|---|']
            for g in run['groups']:
                def pct(x): return '—' if x is None else f'{100*x:.1f}%'
                pairs = ', '.join(f"{p['from']}→{p['to']} {pct(p['share'])}" for p in g['top_pairs'][:2]) or '—'
                docs.append(f"| {g['cluster']} | {g['users']} | {g['median_actions'] if g['median_actions'] is not None else '—'} | "
                            f"{pct(g['event_share'].get('view'))} | {pct(g['users_with_action'].get('cart'))} | "
                            f"{pct(g['users_with_action'].get('purchase'))} | {pct(g['users_with_action'].get('remove'))} | {pairs} |")
            docs.append('')
    docs += ['## 해석 및 논문 수록 시 주의', '',
             '- 표본은 전체 모집단 대표 표본이 아니다. 세션 수에 따른 층화 표본이며 Cosmetics는 사용자 ID modulo 100 표본이다.',
             '- 2개 이상 세션 조건은 사용자 전체 관측 이력을 활용한 사후 조건이다. 온라인 시점에 이 정보를 미리 사용할 수 없다.',
             '- 기존 로더는 session ID 결측 행을 제외한다. 중복·세션 정의 및 64개 토큰 절단도 분포에 영향을 준다.',
             '- 데이터셋별 인코더와 임베딩 공간이 달라 군집 번호나 좌표를 직접 대응시키지 않는다.',
             '- 10-NN 그래프에 연결되지 않은 컴포넌트가 있으면 scikit-learn이 연결을 보완한다. 그래프 군집의 독립성·안정성을 확인한 것은 아니다.',
             '- 사용자 임베딩과 행동 프로파일이 동일 prefix를 사용하므로 차이는 기술적 요약일 뿐 독립 검증 또는 원인 설명이 아니다.',
             '- 그룹을 진짜 선호·만족도·구매 의도나 이탈, uplift 유형으로 해석하지 않는다. 같은 사용자 반복/시간별/인코더 재학습 안정성과 미래 행동 검증은 수행하지 않았다.',
             '- 논문에는 데이터 탐색과 시뮬레이터 동기 부여로 포함할 수 있으나, 표본 선정·토큰화·군집 설정과 한계를 명시하고 가능하면 후속 시점 평가를 추가한다.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(docs), encoding='utf-8')


if __name__ == '__main__':
    for source in sys.argv[1:]:
        if source not in ('electronics', 'cosmetics'):
            raise ValueError(source)
        run(source)
    report()
