#!/usr/bin/env python3
"""
Analyze SQL errors in detail to understand what types of errors are occurring.
"""

import json
import pickle
from collections import defaultdict, Counter
import argparse

def analyze_sql_errors(input_pkl_path, max_samples=None):
    """Analyze SQL errors in detail."""
    
    # Load data
    with open(input_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        samples = []
        for key, entry in data.items():
            candidates = list(entry.get("candidates", []))
            try:
                candidates.sort(key=lambda c: int(c.get("index", 0)))
            except Exception:
                pass
            predicted_sqls = [c.get("sql", "") for c in candidates]
            candidate_exec_results = [c.get("result") for c in candidates]
            gt = entry.get("ground_truth", {}) or {}
            samples.append({
                "sample_id": entry.get("sample_id", key),
                "dataset_name": entry.get("dataset_name", ""),
                "db_id": entry.get("db_id", ""),
                "question": entry.get("question", ""),
                "ground_truth_sql": gt.get("sql", ""),
                "predicted_sqls": predicted_sqls,
                "gt_execution_result": gt.get("result"),
                "candidate_execution_results": candidate_exec_results,
            })
    else:
        samples = data
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Analyzing {len(samples)} samples...")
    
    # Statistics
    total_sqls = 0
    successful_sqls = 0
    syntax_errors = 0
    infrastructure_errors = 0
    empty_sqls = 0
    other_errors = 0
    
    error_breakdown = Counter()
    infrastructure_error_breakdown = Counter()
    
    for sample in samples:
        predicted_sqls = sample.get("predicted_sqls", [])
        candidate_results = sample.get("candidate_execution_results", [])
        
        for i, (sql, result) in enumerate(zip(predicted_sqls, candidate_results)):
            total_sqls += 1
            
            # Check if SQL is empty
            if not sql or not sql.strip():
                empty_sqls += 1
                continue
            
            # Check execution result
            if result is None:
                other_errors += 1
                error_breakdown["No cached result"] += 1
                continue
                
            if not isinstance(result, dict):
                other_errors += 1
                error_breakdown["Invalid result format"] += 1
                continue
            
            if result.get("ok", False):
                successful_sqls += 1
            else:
                error = result.get("error", "").lower()
                
                # Check for infrastructure errors
                infrastructure_indicators = [
                    "timeout", "connection", "network", "http", "request", 
                    "api", "server", "service unavailable", "timed out",
                    "connection refused", "connection timeout"
                ]
                
                if any(indicator in error for indicator in infrastructure_indicators):
                    infrastructure_errors += 1
                    infrastructure_error_breakdown[error[:100]] += 1
                else:
                    syntax_errors += 1
                    error_breakdown[error[:100]] += 1
    
    # Print results
    print("\n" + "="*60)
    print("SQL ERROR ANALYSIS")
    print("="*60)
    print(f"Total SQLs analyzed: {total_sqls}")
    print(f"Successful SQLs: {successful_sqls} ({successful_sqls/total_sqls*100:.2f}%)")
    print(f"Syntax errors: {syntax_errors} ({syntax_errors/total_sqls*100:.2f}%)")
    print(f"Infrastructure errors: {infrastructure_errors} ({infrastructure_errors/total_sqls*100:.2f}%)")
    print(f"Empty SQLs: {empty_sqls} ({empty_sqls/total_sqls*100:.2f}%)")
    print(f"Other errors: {other_errors} ({other_errors/total_sqls*100:.2f}%)")
    
    print(f"\nUnaccounted SQLs: {total_sqls - successful_sqls - syntax_errors - infrastructure_errors - empty_sqls - other_errors}")
    
    print("\n" + "="*60)
    print("TOP SYNTAX ERRORS")
    print("="*60)
    for error, count in error_breakdown.most_common(20):
        print(f"{count:4d} | {error}")
    
    print("\n" + "="*60)
    print("TOP INFRASTRUCTURE ERRORS")
    print("="*60)
    for error, count in infrastructure_error_breakdown.most_common(10):
        print(f"{count:4d} | {error}")
    
    return {
        "total_sqls": total_sqls,
        "successful_sqls": successful_sqls,
        "syntax_errors": syntax_errors,
        "infrastructure_errors": infrastructure_errors,
        "empty_sqls": empty_sqls,
        "other_errors": other_errors,
        "error_breakdown": dict(error_breakdown),
        "infrastructure_error_breakdown": dict(infrastructure_error_breakdown)
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze SQL errors in detail")
    parser.add_argument("--input-pkl", required=True, help="Path to detailed_results.pkl")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to analyze")
    parser.add_argument("--output", help="Output file for detailed results")
    
    args = parser.parse_args()
    
    results = analyze_sql_errors(args.input_pkl, args.max_samples)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

if __name__ == "__main__":
    main()
