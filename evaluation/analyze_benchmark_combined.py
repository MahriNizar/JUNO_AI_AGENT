import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import Counter

Z_95 = 1.959963984540054

def load_benchmark(spec_path):
    with open(spec_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {q['id']: q for q in data}

def is_subseq(sub, full):
    it = iter(full)
    return all(any(x == y for y in it) for x in sub)

def _coverage_score(required_tools, agent_sequence):
    """Return the fraction of required tools invoked, ignoring order."""
    if not required_tools:
        return 0.0

    required = Counter(required_tools)
    observed = Counter(agent_sequence)
    matched = sum(min(observed[tool], count) for tool, count in required.items())
    total = sum(required.values())
    return float(matched) / total if total else 0.0

def compute_tsr(agent_data, query_spec):
    if agent_data.get('crashed', False):
        return 0.0
        
    tool_calls = agent_data.get('tool_calls', [])
    agent_sequence = [tc['skill_name'] for tc in tool_calls]

    allowed_refusals = query_spec.get('allowed_refusals')
    if allowed_refusals is not None:
        return 1.0 if len(tool_calls) == 0 else 0.0

    expected_tools = query_spec.get('expected_tools') or []
    accepted_toolchains = query_spec.get('accepted_toolchains')
    if accepted_toolchains:
        candidate_chains = list(accepted_toolchains)
        if expected_tools:
            candidate_chains.append(expected_tools)

        return max(_coverage_score(chain, agent_sequence) for chain in candidate_chains)

    if expected_tools:
        alternative_tools = query_spec.get('alternative_tools') or []
        if len(expected_tools) == 1:
            valid_tools_set = set(expected_tools).union(alternative_tools)
            return 1.0 if any(tool in valid_tools_set for tool in agent_sequence) else 0.0

        return 1.0 if is_subseq(expected_tools, agent_sequence) else 0.0
        
    return 0.0

def compare_args(agent_args, expected_args):
    if not expected_args:
        return 1.0
    matches = 0
    for k, v in expected_args.items():
        if k in agent_args and str(agent_args[k]) == str(v):
            matches += 1
    return float(matches) / len(expected_args)

def compute_acr(agent_data, query_spec, tsr):
    if tsr == 0.0:
        return None
        
    expected_args = query_spec.get('expected_args')
    if not expected_args:
        return None
        
    tool_calls = agent_data.get('tool_calls', [])
    
    if isinstance(expected_args, dict):
        # single dict
        target_tools = set(query_spec.get('expected_tools', []) + (query_spec.get('alternative_tools') or []))
        agent_tc = None
        for tc in tool_calls:
            if tc['skill_name'] in target_tools:
                agent_tc = tc
                break
        if not agent_tc and tool_calls:
            agent_tc = tool_calls[0]
            
        if not agent_tc:
            return 0.0
        return compare_args(agent_tc.get('args', {}), expected_args)
        
    elif isinstance(expected_args, list):
        scores = []
        for exp_arg_entry in expected_args:
            if not isinstance(exp_arg_entry, dict): continue
            for tool_name, exp_kwargs in exp_arg_entry.items():
                valid_names = set([tool_name] + (query_spec.get('alternative_tools') or []))
                agent_tc = None
                for tc in tool_calls:
                    if tc['skill_name'] in valid_names:
                        agent_tc = tc
                        break
                if not agent_tc:
                    scores.append(0.0)
                else:
                    scores.append(compare_args(agent_tc.get('args', {}), exp_kwargs))
        if not scores:
            return None
        return sum(scores) / len(scores)

    return None

def wilson_errors(rate, n, z=Z_95):
    """Return asymmetric Wilson 95% CI errors for a bounded rate in [0, 1]."""
    if n <= 0 or pd.isna(rate):
        return 0.0, 0.0

    p = min(1.0, max(0.0, float(rate)))
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return max(0.0, p - lower), max(0.0, upper - p)

def main():
    bench_file = "evaluation/query_benchmark.json"
    results_file_1 = "evaluation/results/valid_tests/t1_t2_t5_5times.json"
    results_file_2 = "evaluation/results/valid_tests/t3_comp_5times.json"
    
    if not os.path.exists(bench_file) or not os.path.exists(results_file_1) or not os.path.exists(results_file_2):
        print("Required files not found. Run from the evaluation directory.")
        sys.exit(1)
        
    benchmark = load_benchmark(bench_file)
    with open(results_file_1, 'r', encoding='utf-8') as f:
        eval_data_1 = json.load(f)
    with open(results_file_2, 'r', encoding='utf-8') as f:
        eval_data_2 = json.load(f)
        
    results = eval_data_1.get('results', []) + eval_data_2.get('results', [])
    
    data_rows = []
    for run_result in results:
        query_id = run_result.get('query_id')
        tier = run_result.get('tier')
        run_id = run_result.get('run_id')
        
        spec = benchmark.get(query_id)
        if not spec: continue
        
        for agent in ['juno', 'baseline_b1']:
            if agent not in run_result: continue
            agent_data = run_result[agent]
            
            crashed = agent_data.get('crashed', False)
            tool_call_count = agent_data.get('tool_call_count', 0)
            
            tsr = compute_tsr(agent_data, spec)
            acr = compute_acr(agent_data, spec, tsr)
            
            data_rows.append({
                'query_id': query_id,
                'tier': tier,
                'run_id': run_id,
                'agent_name': agent,
                'tsr': tsr,
                'acr': acr,
                'crashed': float(crashed),
                'tool_call_count': tool_call_count
            })
            
    df = pd.DataFrame(data_rows)
    
    # ---------------- SUMMARY TABLE ----------------
    print(f"{'Tier':<6} | {'Agent':<12} | {'Mean TSR ± std':<20} | {'Mean ACR ± std':<20} | {'Crash Rate':<12} | {'QueriesxRuns':<12}")
    print("-" * 90)
    
    agents = ['juno', 'baseline_b1']
    tiers = sorted(df['tier'].unique())
    
    for t in tiers:
        for agent in agents:
            sub_df = df[(df['tier'] == t) & (df['agent_name'] == agent)]
            if len(sub_df) == 0: continue
            
            tsr_mean = sub_df['tsr'].mean()
            tsr_std = sub_df['tsr'].std()
            acr_mean = sub_df['acr'].dropna().mean()
            if pd.isna(acr_mean):
                acr_mean_str = "N/A"
            else:
                acr_std = sub_df['acr'].dropna().std()
                acr_mean_str = f"{acr_mean:.3f} ± {acr_std:.3f}"
                
            crash_rt = sub_df['crashed'].mean()
            count = len(sub_df)
            
            print(f"{t:<6} | {agent:<12} | {tsr_mean:.3f} ± {tsr_std:.3f} | {acr_mean_str:<20} | {crash_rt:.3f}      | {count:<12}")
            
    # Overall summary
    print("-" * 90)
    for agent in agents:
        sub_df = df[df['agent_name'] == agent]
        tsr_mean = sub_df['tsr'].mean()
        tsr_std = sub_df['tsr'].std()
        acr_mean = sub_df['acr'].dropna().mean()
        if pd.isna(acr_mean):
            acr_mean_str = "N/A"
        else:
            acr_std = sub_df['acr'].dropna().std()
            acr_mean_str = f"{acr_mean:.3f} ± {acr_std:.3f}"
        crash_rt = sub_df['crashed'].mean()
        count = len(sub_df)
        print(f"{'All':<6} | {agent:<12} | {tsr_mean:.3f} ± {tsr_std:.3f} | {acr_mean_str:<20} | {crash_rt:.3f}      | {count:<12}")

    # ---------------- PLOTTING ----------------
    plt.style.use('default')
    
    # Plot 1: Headline Comparison
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    groups = ['TSR', 'ACR', 'Crash Rate']
    x = np.arange(len(groups))
    width = 0.35
    
    juno_means = []
    juno_errs_lower = []
    juno_errs_upper = []
    base_means = []
    base_errs_lower = []
    base_errs_upper = []
    
    for metric in ['tsr', 'acr', 'crashed']:
        for agent, means, errs_low, errs_up in [
            ('juno', juno_means, juno_errs_lower, juno_errs_upper),
            ('baseline_b1', base_means, base_errs_lower, base_errs_upper)
        ]:
            sub = df[df['agent_name'] == agent]
            if metric == 'acr':
                vals = sub[metric].dropna()
            else:
                vals = sub[metric]
            
            mean_val = vals.mean()
            n = len(vals)
            err_low, err_up = wilson_errors(mean_val, n)
            
            means.append(mean_val)
            errs_low.append(err_low)
            errs_up.append(err_up)
            
    rects1 = ax1.bar(
        x - width/2, juno_means, width,
        yerr=[juno_errs_lower, juno_errs_upper],
        label='JUNO', capsize=5, color='#4C72B0'
    )
    rects2 = ax1.bar(
        x + width/2, base_means, width,
        yerr=[base_errs_lower, base_errs_upper],
        label='Baseline', capsize=5, color='#DD8452'
    )
    
    ax1.set_ylabel('Mean Value')
    ax1.set_title('Overall Agent Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    
    # Value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height):
                ax1.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
    autolabel(rects1)
    autolabel(rects2)
    
    # Minimal chartjunk
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    fig1.tight_layout()
    p1_name = 'plot_headline_comparison.png'
    fig1.savefig(p1_name)
    print(f"Saved {os.path.abspath(p1_name)}")
    
    # Plot 2: Performance (ACR) by Tier (Bar Chart excluding Tier 5)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    juno_tier_means = []
    juno_tier_errs_lower = []
    juno_tier_errs_upper = []
    base_tier_means = []
    base_tier_errs_lower = []
    base_tier_errs_upper = []
    
    valid_tiers_p2 = [t for t in sorted(df['tier'].unique()) if t != 5]
    x_pos = np.arange(len(valid_tiers_p2))
    width = 0.35
    
    for t in valid_tiers_p2:
        sub_j = df[(df['tier'] == t) & (df['agent_name'] == 'juno')]['acr'].dropna()
        sub_b = df[(df['tier'] == t) & (df['agent_name'] == 'baseline_b1')]['acr'].dropna()
        
        n_j = len(sub_j)
        mean_j = sub_j.mean() if n_j > 0 else 0
        err_j_low, err_j_up = wilson_errors(mean_j, n_j)
        juno_tier_means.append(mean_j)
        juno_tier_errs_lower.append(err_j_low)
        juno_tier_errs_upper.append(err_j_up)
        
        n_b = len(sub_b)
        mean_b = sub_b.mean() if n_b > 0 else 0
        err_b_low, err_b_up = wilson_errors(mean_b, n_b)
        base_tier_means.append(mean_b)
        base_tier_errs_lower.append(err_b_low)
        base_tier_errs_upper.append(err_b_up)
        
    rects_j2 = ax2.bar(
        x_pos - width/2, juno_tier_means, width,
        yerr=[juno_tier_errs_lower, juno_tier_errs_upper],
        label='JUNO', capsize=5, color='#4C72B0'
    )
    rects_b2 = ax2.bar(
        x_pos + width/2, base_tier_means, width,
        yerr=[base_tier_errs_lower, base_tier_errs_upper],
        label='Baseline', capsize=5, color='#DD8452'
    )
    
    ax2.set_xlabel('Tier')
    ax2.set_ylabel('Mean ACR')
    ax2.set_title('Argument Correctness Rate by Tier')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(valid_tiers_p2)
    ax2.set_ylim(0, 1.15)
    ax2.legend()
    
    # Value labels
    def autolabel_acr(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax2.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
                            
    autolabel_acr(rects_j2)
    autolabel_acr(rects_b2)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig2.tight_layout()
    p2_name = 'plot_acr_by_tier.png'
    fig2.savefig(p2_name)
    print(f"Saved {os.path.abspath(p2_name)}")
    
    # Plot 3 (revised): Per-Tier Success Rate Bar Chart
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    juno_succ_rates = []
    juno_errs_lower = []
    juno_errs_upper = []
    juno_labels = []
    
    base_succ_rates = []
    base_errs_lower = []
    base_errs_upper = []
    base_labels = []
    
    valid_tiers_p3 = sorted(df['tier'].unique())
    for t in valid_tiers_p3:
        for agent, succ_rates, errs_low, errs_up, labels in [
            ('juno', juno_succ_rates, juno_errs_lower, juno_errs_upper, juno_labels),
            ('baseline_b1', base_succ_rates, base_errs_lower, base_errs_upper, base_labels)
        ]:
            sub = df[(df['tier'] == t) & (df['agent_name'] == agent)]
            n = len(sub)
            if n == 0:
                succ_rates.append(0.0)
                errs_low.append(0.0)
                errs_up.append(0.0)
                labels.append('0/0')
                continue
                
            k = sum(sub['tsr'] == 1.0)
            p = k / n
            
            err_low, err_up = wilson_errors(p, n)
            succ_rates.append(p)
            errs_low.append(err_low)
            errs_up.append(err_up)
            labels.append(f'{k}/{n}')
            
    x_pos = np.arange(len(valid_tiers_p3))
    width = 0.35
    
    rects_j = ax3.bar(x_pos - width/2, juno_succ_rates, width, yerr=[juno_errs_lower, juno_errs_upper], label='JUNO', capsize=5, color='#4C72B0')
    rects_b = ax3.bar(x_pos + width/2, base_succ_rates, width, yerr=[base_errs_lower, base_errs_upper], label='Baseline', capsize=5, color='#DD8452')
    
    ax3.set_xlabel('Tier')
    ax3.set_ylabel('Success Rate (TSR=1.0)')
    ax3.set_title('Per-Tier Success Rate')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(valid_tiers_p3)
    ax3.set_ylim(0, 1.1)
    ax3.legend()
    
    # Value labels
    def autolabel_succ(rects, labels_text, errs_up):
        for rect, text, err in zip(rects, labels_text, errs_up):
            height = rect.get_height()
            ax3.annotate(text,
                        xy=(rect.get_x() + rect.get_width() / 2, height + err),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    autolabel_succ(rects_j, juno_labels, juno_errs_upper)
    autolabel_succ(rects_b, base_labels, base_errs_upper)
    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    fig3.tight_layout()
    p3_name = 'plot_success_rate_by_tier.png'
    fig3.savefig(p3_name)
    print(f"Saved {os.path.abspath(p3_name)}")

if __name__ == "__main__":
    main()
