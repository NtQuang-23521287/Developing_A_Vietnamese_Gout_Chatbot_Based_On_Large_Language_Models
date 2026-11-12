#!/usr/bin/env python3
"""Rebuild baseline figures from Judge v2 results for the Gout KLTN.

Run from repository root:
  python scripts/plot_baseline_results_v2.py --split all
  python scripts/plot_baseline_results_v2.py --split test

Read either artifacts/runs/<run>/eval_v2/{summary.json,scores.jsonl}
or artifacts/runs/<run>/{summary.json,scores.jsonl} (the current zipped export).
No OpenAI API or GPU is needed.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = {
    'Qwen3-8B': '20261025T074006Z_baseline_qwen3_8b_7963',
    'Gemma3-4B': '20261025T080005Z_baseline_gemma3_4b_5283',
    'SeaLLMs-v3-7B': '20261025T080440Z_baseline_seallms_v3_7b_5915',
}
QUALITY = ['faithfulness', 'completeness', 'relevance', 'context_recall',
           'patient_utility', 'refusal_appropriateness']
RISK = ['hallucination_level', 'omission_risk', 'safety_risk']
RATE_KEYS = ['hallucination_level>=2', 'omission_risk>=2',
             'safety_risk>=2', 'safety_risk>=3']
QTYPE = ['basic_knowledge', 'clinical_reasoning', 'red_flag', 'safety_trap']
RISK_LEVEL = ['low', 'moderate', 'high', 'critical']


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def wavg(values, weights):
    a = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(a) & np.isfinite(w)
    return float(np.average(a[mask], weights=w[mask])) if mask.any() else float('nan')


def boot_ci(values, weights, n=5000, seed=2026):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    numer = np.sum(values[idx] * weights[idx], axis=1)
    denom = np.sum(weights[idx], axis=1)
    return tuple(float(x) for x in np.quantile(numer / denom, [0.025, 0.975]))


def load_run(root, model, run_name, split):
    run_dir = root / 'artifacts' / 'runs' / run_name
    for folder in (run_dir / 'eval_v2', run_dir):
        if (folder / 'scores.jsonl').is_file() and (folder / 'summary.json').is_file():
            break
    else:
        raise FileNotFoundError(f'No Judge v2 scores for {model}: {run_dir}')
    summary = load_json(folder / 'summary.json')
    if summary.get('judge_version') != 'judge_vi_v2' or summary.get('reward_version') != 'heuristic_reward_v2':
        raise ValueError(f'{model}: found non-v2 summary, refuse to plot: {folder}')
    rows = load_jsonl(folder / 'scores.jsonl')
    if len(rows) != len(set(r['case_id'] for r in rows)):
        raise ValueError(f'{model}: duplicated case_id in scores')
    if any(r['status'] != 'ok' for r in rows):
        raise ValueError(f'{model}: some judge rows failed')
    if summary.get('n_scored') != len(rows):
        raise ValueError(f'{model}: summary count != scores count')
    if split != 'all':
        rows = [r for r in rows if r['split'] == split]
    if not rows:
        raise ValueError(f'{model}: no rows for split={split}')
    preds_path = run_dir / 'predictions.jsonl'
    if not preds_path.is_file():
        raise FileNotFoundError(f'{model}: missing predictions for latency: {preds_path}')
    preds = load_jsonl(preds_path)
    ids = {r['case_id'] for r in rows}
    preds = [p for p in preds if p['case_id'] in ids]
    if any(p['status'] != 'ok' for p in preds):
        raise ValueError(f'{model}: prediction error found')
    latency = float(np.mean([float(p['latency_seconds']) for p in preds]))
    return {
        'rows': {r['case_id']: r for r in rows},
        'summary': summary,
        'judge_dir': str(folder),
        'latency': latency,
        'n_turns': len(preds),
    }


def summarize(data, split):
    rows = list(data['rows'].values())
    weights = np.array([r['risk_weight'] for r in rows], dtype=float)
    rewards = np.array([r['result']['reward'] for r in rows], dtype=float)
    if split == 'all':
        s = data['summary']
        computed = wavg(rewards, weights)
        if abs(computed - s['reward_weighted_mean']) > 1e-10:
            raise ValueError('Summary reward disagrees with per-case scores')
        ci = s['reward_weighted_ci95']
    else:
        ci = boot_ci(rewards, weights)
    return {
        'n': len(rows),
        'n_turns': data['n_turns'],
        'reward': wavg(rewards, weights),
        'ci_lo': float(ci[0]), 'ci_hi': float(ci[1]),
        'reward_unweighted': float(np.mean(rewards)),
        'latency': data['latency'],
        'fatal_count': int(sum(r['result']['fatal'] for r in rows)),
        'quality': {k: wavg([r['result']['M'].get(k, np.nan) for r in rows], weights) for k in QUALITY},
        'risk': {k: wavg([r['raw'][k] for r in rows], weights) for k in RISK},
        'rates': {k: wavg([float(r['raw'][k.split('>=')[0]]) >= int(k.split('>=')[1]) for r in rows], weights)
                  for k in RATE_KEYS},
        'by_type': {k: (wavg([r['result']['reward'] for r in rows if r['question_type']==k],
                            [r['risk_weight'] for r in rows if r['question_type']==k])
                        if any(r['question_type']==k for r in rows)
                        else float('nan')) for k in QTYPE},
        'by_risk': {k: (wavg([r['result']['reward'] for r in rows if r['risk_level']==k],
                            [r['risk_weight'] for r in rows if r['risk_level']==k])
                        if any(r['risk_level']==k for r in rows)
                        else float('nan')) for k in RISK_LEVEL},
        'n_type': {k: sum(r['question_type']==k for r in rows) for k in QTYPE},
        'n_risk': {k: sum(r['risk_level']==k for r in rows) for k in RISK_LEVEL},
    }


def save(fig, outdir, fname):
    fig.tight_layout()
    fig.savefig(outdir / f'{fname}.png', dpi=300, bbox_inches='tight')
    fig.savefig(outdir / f'{fname}.svg', bbox_inches='tight')
    plt.close(fig)


def grouped_bars(models, labels, value_fn, ylabel, title, outdir, fname, ylim=None, label_values=False,
                 xticklabels=None):
    n = len(labels)
    x = np.arange(n)
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(9, n*1.8), 5.8))
    for mi, model in enumerate(models):
        vals = np.asarray([value_fn(model, k) for k in labels], dtype=float)
        bar = ax.bar(x + (mi-1)*width, vals, width, label=model)
        if label_values:
            for rect, val in zip(bar, vals, strict=True):
                if np.isfinite(val):
                    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f'{val:.2f}',
                            ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x, xticklabels if xticklabels is not None else labels, rotation=12, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    save(fig,outdir,fname)


def plot_all(root, outdir, split, n_boot=5000):
    dataset = {m:load_run(root, m, ident, split) for m,ident in RUNS.items()}
    signatures = {m:(d['summary']['testset_sha256'],d['summary']['judge'],
                     d['summary']['judge_version'],d['summary']['reward_version']) for m,d in dataset.items()}
    if len(set(signatures.values())) != 1:
        raise ValueError(f'Different testsets / Judge models / versions: {signatures}')
    model_order = list(RUNS)
    ids = [set(dataset[m]['rows']) for m in model_order]
    if len(set(map(frozenset,ids))) != 1:
        raise ValueError('Three models do not share identical case_id sets')
    for cid in ids[0]:
        base = dataset[model_order[0]]['rows'][cid]
        if any(dataset[m]['rows'][cid]['risk_weight'] != base['risk_weight'] or
               dataset[m]['rows'][cid]['question_type'] != base['question_type'] or
               dataset[m]['rows'][cid]['risk_level'] != base['risk_level'] for m in model_order):
            raise ValueError(f'Inconsistent labels across models: {cid}')
    sums = {m:summarize(dataset[m],split) for m in model_order}
    counts = {m:sums[m]['n'] for m in model_order}
    if len(set(counts.values())) != 1:
        raise ValueError(f'Different case counts: {counts}')
    n = next(iter(counts.values()))
    outdir.mkdir(parents=True, exist_ok=True)

    # Chart 1: reward weighted by risk level; 95% bootstrap CIs.
    rewards = [sums[m]['reward'] for m in model_order]
    lo = [sums[m]['reward']-sums[m]['ci_lo'] for m in model_order]
    hi = [sums[m]['ci_hi']-sums[m]['reward'] for m in model_order]
    fig,ax=plt.subplots(figsize=(9,5.8))
    bars=ax.bar(model_order,rewards,yerr=np.asarray([lo,hi]),capsize=6)
    for b,m,v in zip(bars,model_order,rewards,strict=True):
        ax.text(b.get_x()+b.get_width()/2, sums[m]['ci_hi']+0.016, f'{v:.4f}',ha='center',fontsize=10)
    ax.set_ylim(0, min(1.04,max(sums[m]['ci_hi'] for m in model_order)+0.13))
    ax.set_ylabel('Risk-weighted mean Reward v2')
    ax.set_title(f'Overall Reward v2 (n={n} cases; 95% bootstrap CI)')
    ax.grid(axis='y',alpha=0.25)
    ax.set_axisbelow(True)
    save(fig,outdir,'01_overall_reward')

    # Chart 2: 6 quality metrics normalized from 1..5 to 0..1.
    qlabels=['Faithfulness','Completeness','Relevance','Context Recall',
             'Patient Utility','Refusal\n(Safety Trap only)']
    grouped_bars(model_order, QUALITY,
        lambda m,k:sums[m]['quality'][k],
        'Risk-weighted mean normalized score (0–1)',
        'Quality Metrics M1–M6 (higher is better)',outdir,'02_quality_metrics',ylim=(0,1.07), xticklabels=qlabels)

    # Chart 3: three risk metrics still on RAW 0-3 scale; do not mix with normalized M.
    grouped_bars(model_order, RISK,
        lambda m,k:sums[m]['risk'][k],
        'Risk-weighted mean raw score (0–3)',
        'Clinical Risk P1–P3 (lower is better)',outdir,'03_risk_metrics',ylim=(0,3.12))

    # Chart 4: latency per TURN, not per case, from predictions.jsonl.
    fig,ax=plt.subplots(figsize=(9,5.8))
    for m in model_order:
        ax.scatter(sums[m]['latency'], sums[m]['reward'], s=100, label=m)
        ax.annotate(m,(sums[m]['latency'],sums[m]['reward']),xytext=(8,8),textcoords='offset points')
    ax.set_xlabel('Mean generation latency (seconds per turn)')
    ax.set_ylabel('Risk-weighted mean Reward v2')
    ax.set_title(f'Reward vs. Generation Latency ({split} cases)')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig,outdir,'04_reward_vs_latency')

    # Chart 5 replaces old disease groups with actual v03 question_type.
    xtypes=[k for k in QTYPE if sums[model_order[0]]['n_type'][k]]
    groups=[f'{x.replace("_", " ").title()}\n(n={sums[model_order[0]]["n_type"][x]})' for x in xtypes]
    grouped_bars(model_order, xtypes,lambda m,k:sums[m]['by_type'][k],
                 'Risk-weighted mean Reward v2',
                 'Reward by Question Type (v03)',outdir,'05_reward_by_question_type',ylim=(0,1.08), xticklabels=groups)

    # Chart 6: negative reward is valid. Preserve negative tail; 1.0 max.
    fig,ax=plt.subplots(figsize=(9,5.8))
    data=[[r['result']['reward'] for r in dataset[m]['rows'].values()] for m in model_order]
    ax.boxplot(data, tick_labels=model_order, showmeans=True)
    ax.axhline(0,linestyle=':',linewidth=1)
    ax.set_ylabel('Per-case Reward v2 (can be negative)')
    ax.set_title(f'Reward Distribution by Model (n={n} cases)')
    ax.grid(axis='y',alpha=0.25)
    save(fig,outdir,'06_reward_distribution')

    # Chart 7: clinical risk threshold rates, weighted, separate from P mean values.
    grouped_bars(model_order, RATE_KEYS,lambda m,k:100*sums[m]['rates'][k],
                 'Risk-weighted share of cases (%)',
                 'Clinical Risk Threshold Rates (lower is better)',outdir,'07_risk_threshold_rates',ylim=(0,100),
                 xticklabels=['Hallucination >= 2','Omission >= 2','Safety >= 2','Safety >= 3'])

    # Chart 8: compare identical questions case-by-case (paired weighted bootstrap).
    pairs=list(itertools.combinations(model_order,2))
    sorted_ids=sorted(ids[0])
    comparison=[]
    fig,ax=plt.subplots(figsize=(9,5.8))
    for i,(a,b) in enumerate(pairs):
        ar=np.array([dataset[a]['rows'][cid]['result']['reward'] for cid in sorted_ids])
        br=np.array([dataset[b]['rows'][cid]['result']['reward'] for cid in sorted_ids])
        w=np.array([dataset[a]['rows'][cid]['risk_weight'] for cid in sorted_ids])
        dif=ar-br
        mean=wavg(dif,w)
        ci_lo,ci_hi=boot_ci(dif,w,n=n_boot,seed=2026)
        try:
            from scipy.stats import wilcoxon
            pval=float(wilcoxon(ar,br).pvalue)
        except (ImportError,ValueError):
            pval=float('nan')
        comparison.append({'Model A':a,'Model B':b,'Mean difference (A-B)':mean,
                           'Paired CI95 lower':ci_lo,'Paired CI95 upper':ci_hi,
                           'Wilcoxon p (unweighted)':pval,
                           'A wins':int((dif>0).sum()),'B wins':int((dif<0).sum()),
                           'Ties':int((dif==0).sum()),'N paired':len(dif)})
        ax.errorbar(mean,i,xerr=[[mean-ci_lo],[ci_hi-mean]],fmt='o',capsize=5)
    ax.axvline(0,color='gray',linewidth=1,linestyle='--')
    ax.set_yticks(np.arange(len(pairs)),[f'{a} − {b}' for a,b in pairs])
    ax.set_xlabel('Risk-weighted mean Reward difference (95% paired bootstrap CI)')
    ax.set_title(f'Paired Reward Differences (n={n} shared cases)')
    ax.grid(axis='x',alpha=0.25)
    ax.invert_yaxis()
    save(fig,outdir,'08_paired_reward_differences')

    # Chart 9: differences by clinical risk level.
    risktypes=[k for k in RISK_LEVEL if sums[model_order[0]]['n_risk'][k]]
    grouped_bars(model_order,risktypes,lambda m,k:sums[m]['by_risk'][k],
        'Risk-weighted mean Reward v2', 'Reward by Clinical Risk Level',
        outdir,'09_reward_by_risk_level',ylim=(-0.1,1.05),
        xticklabels=[f'{x.title()} (n={sums[model_order[0]]["n_risk"][x]})' for x in risktypes])

    tab=[]
    for m in model_order:
        s=sums[m]
        d={'Model':m,'Cases':s['n'],'Generation turns':s['n_turns'],
           'Reward weighted':s['reward'],'CI95 low':s['ci_lo'],'CI95 high':s['ci_hi'],
           'Reward unweighted':s['reward_unweighted'],'Fatal cases':s['fatal_count'],
           'Mean latency seconds per turn':s['latency']}
        d.update({'M_'+k:s['quality'][k] for k in QUALITY})
        d.update({'P_'+k+'_raw_0_3':s['risk'][k] for k in RISK})
        d.update({'Rate_'+k:s['rates'][k] for k in RATE_KEYS})
        tab.append(d)
    pd.DataFrame(tab).to_csv(outdir/'baseline_model_summary_v2.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(comparison).to_csv(outdir/'pairwise_reward_comparison_v2.csv',index=False,encoding='utf-8-sig')
    metadata={
        'split':split,'cases':n,'models':RUNS,'judge':dataset[model_order[0]]['summary']['judge'],
        'judge_version':'judge_vi_v2','reward_version':'heuristic_reward_v2',
        'testset_sha256':dataset[model_order[0]]['summary']['testset_sha256'],
        'methodology':{
            'reward':'Risk-level-weighted mean, not simple mean.',
            'confidence_interval':('For all use original judge summary CI95; '
                                   'for split use 5000 percentile bootstrap resamples (seed 2026).'),
            'quality':'Normalized M=(raw_score-1)/4, risk-weighted.',
            'risk':'Raw 0..3 risk metric, risk-weighted.',
            'rates':'Risk-level weighted case rate with threshold.',
            'latency':'Unweighted mean generation latency per turn from predictions.jsonl (GPU/device-dependent).',
            'comparison':'Paired bootstrap difference weighted by risk_level, Wilcoxon p unweighted.',
            'caveat':'Includes data-driven model judge ratings, not clinician-validated medical safety.'}}
    (outdir/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'PASS: generated 9 charts PNG+SVG and 2 tables under {outdir} (n={n})')
    tab_cols = ['Model','Cases','Reward weighted','CI95 low','CI95 high','Fatal cases','Mean latency seconds per turn']
    print(pd.DataFrame(tab)[tab_cols].to_string(index=False))
    cmp_cols = ['Model A','Model B','Mean difference (A-B)','Paired CI95 lower','Paired CI95 upper',
                'Wilcoxon p (unweighted)']
    print(pd.DataFrame(comparison)[cmp_cols].to_string(index=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path('.'))
    p.add_argument('--split',choices=['all','test','dev'],default='all')
    p.add_argument('--out-dir',type=Path,default=None)
    args=p.parse_args()
    root=args.repo_root.resolve()
    outdir=args.out_dir or root / 'artifacts' / 'analysis_charts_v2' / args.split
    plot_all(root,outdir,args.split)

if __name__=='__main__':
    main()
