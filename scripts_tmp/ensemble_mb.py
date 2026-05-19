"""Build SOTA v5 multi-basho ensemble from honestly-trained streams.

Each stream trained on bashoId < 202307, val predictions on {202307,202309,202311}.
Multi-basho val has ~900 rows (~3x current). SE on val_AUC ~halved.

Selection rule: each weight must keep val_acc, val_AUC, val_LL all strictly
above the bag20_lucky_mb_iso baseline (the new dominant single stream).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

# Load all multi-basho streams
streams = {
    'base_mb':   np.load('runs/bag20_lucky_mb/probs.npz'),
    'v4_mb':     np.load('runs/bag_diverse_v4_mb/probs.npz'),
    'lr_mb':     np.load('runs/aux_mb_lr.npz'),
    'mlp_mb':    np.load('runs/aux_mb_mlp.npz'),
    'mlp10_mb':  np.load('runs/aux_mb_mlp10.npz'),
    'cb_mb':     np.load('runs/aux_mb_cb.npz'),
    'sd10_mb':   np.load('runs/siamese_mb_probs.npz'),
}
which_v = {n: 'val_iso' for n in streams}
which_t = {n: 'test_iso' for n in streams}

y_va = streams['base_mb']['y_val']
y_te = streams['base_mb']['y_test']

print(f'Multi-basho val: {len(y_va)} rows, test: {len(y_te)} rows')
print(f'\nIndividual streams (iso):')
for n in streams:
    v = streams[n][which_v[n]]
    t = streams[n][which_t[n]]
    print(f'  {n:10s}  v_acc={accuracy_score(y_va, v>0.5):.4f}  v_auc={roc_auc_score(y_va, v):.4f}  v_ll={log_loss(y_va, np.clip(v, 1e-6, 1-1e-6)):.4f}  |  t_acc={accuracy_score(y_te, t>0.5):.4f}  t_auc={roc_auc_score(y_te, t):.4f}')

names = list(streams.keys())
V = np.stack([streams[n][which_v[n]] for n in names])
T = np.stack([streams[n][which_t[n]] for n in names])

# Baseline = strongest single stream by val_AUC (multi-basho-honest)
val_aucs_indiv = np.array([roc_auc_score(y_va, V[i]) for i in range(len(names))])
i_best = np.argmax(val_aucs_indiv)
B_NAME = names[i_best]
B_V = V[i_best]; B_T = T[i_best]
B_VA = accuracy_score(y_va, B_V>0.5)
B_VAU = val_aucs_indiv[i_best]
B_VLL = log_loss(y_va, np.clip(B_V, 1e-6, 1-1e-6))
B_TA = accuracy_score(y_te, B_T>0.5)
print(f'\nBaseline = {B_NAME}: v_acc={B_VA:.4f} v_auc={B_VAU:.4f} v_ll={B_VLL:.4f} test_acc={B_TA:.4f}')

# Convex-grid search step 0.05 over 7 streams
def all_w(n, step):
    k = int(round(1.0/step))
    out = []
    def rec(rem, slots, prefix):
        if slots == 1: out.append(prefix + (rem*step,)); return
        for i in range(rem+1): rec(rem-i, slots-1, prefix + (i*step,))
    rec(k, n, ())
    return out

# Step 0.1 for 7 streams = 8008 configs (manageable)
configs = np.array(all_w(7, 0.1))
print(f'\nSearching {len(configs)} configs (step=0.1)...')
v_probs = configs @ V
t_probs = configs @ T
val_accs = np.mean((v_probs>0.5).astype(int) == y_va, axis=1)
val_aucs = np.array([roc_auc_score(y_va, v_probs[i]) for i in range(len(configs))])
val_lls = np.array([log_loss(y_va, np.clip(v_probs[i], 1e-6, 1-1e-6)) for i in range(len(configs))])
test_accs = np.mean((t_probs>0.5).astype(int) == y_te, axis=1)
test_aucs = np.array([roc_auc_score(y_te, t_probs[i]) for i in range(len(configs))])

# Honest: all 3 val criteria > baseline
mask = (val_accs > B_VA - 1e-9) & (val_aucs > B_VAU + 1e-6) & (val_lls < B_VLL - 1e-6)
print(f'\nHonest configs (val_acc≥, val_AUC>, val_LL<): {mask.sum()}')
if mask.sum() > 0:
    idx = np.where(mask)[0]
    # Top by val_AUC
    sorted_idx = idx[np.argsort(-val_aucs[idx])]
    print(f'\nTop 10 by val_AUC among honest:')
    print(f'{"weights":<90} {"v_acc":>7} {"v_auc":>7} {"v_ll":>7}  {"t_acc":>7} {"t_auc":>7}')
    for i in sorted_idx[:10]:
        w = configs[i]
        wd = ' '.join(f'{names[j][:6]}={round(w[j],2)}' for j in range(len(names)) if w[j] > 0.005)
        print(f'  {wd:<88} {val_accs[i]:>7.4f} {val_aucs[i]:>7.4f} {val_lls[i]:>7.4f}  {test_accs[i]:>7.4f} {test_aucs[i]:>7.4f}')

    # Best by test acc among honest (oracle, for diagnosis)
    sorted_idx_t = idx[np.argsort(-test_accs[idx])]
    print(f'\nTop 5 by test_acc among honest (oracle, diagnostic):')
    for i in sorted_idx_t[:5]:
        w = configs[i]
        wd = ' '.join(f'{names[j][:6]}={round(w[j],2)}' for j in range(len(names)) if w[j] > 0.005)
        print(f'  {wd:<88} {val_accs[i]:>7.4f} {val_aucs[i]:>7.4f} {val_lls[i]:>7.4f}  {test_accs[i]:>7.4f} {test_aucs[i]:>7.4f}')

    # Save the val_AUC-best honest as SOTA v5
    i_pick = sorted_idx[0]
    w_pick = configs[i_pick]
    val_final = w_pick @ V
    test_final = w_pick @ T
    np.savez('runs/sota_v5_mb_probs.npz',
             val_iso=val_final, test_iso=test_final,
             y_val=y_va, y_test=y_te,
             weights={n: float(w_pick[i]) for i, n in enumerate(names)})
    print(f'\nSaved SOTA v5 mb: t_acc={test_accs[i_pick]:.4f}')
    print(f'  weights: {dict(zip(names, w_pick))}')

# Compare to v4.8 (single-basho honest, evaluated on same test)
print(f'\n=== Reference comparison ===')
print(f'v4.8 (single-basho val, train<202311): test_acc=0.6147 test_auc=0.6425')
print(f'   But that v4.8 was selected using val=202311 only — val signal more noisy.')
print(f'SOTA v5 (multi-basho val, train<202307): test_acc above^^^')
EOF
