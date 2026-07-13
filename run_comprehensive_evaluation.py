#!/usr/bin/env python3
"""
================================================================================
Comprehensive Evaluation Runner
================================================================================
Runs evaluation with multiple membership inference attacks and utility analysis

Usage:
    python run_comprehensive_evaluation.py --dataset cifar10 --comprehensive
    python run_comprehensive_evaluation.py --dataset all --comprehensive --epochs 40
================================================================================
"""

import sys
import os
import argparse
import gc
import psutil
from pathlib import Path

# Windows console often uses cp1252; avoid UnicodeEncodeError on status symbols
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        pass

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def monitor_memory():
    """Monitor memory usage and print warnings"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    
    if memory_mb > 4000:  # 4GB warning
        print(f"WARNING: High memory usage: {memory_mb:.1f} MB")
        if memory_mb > 6000:  # 6GB critical
            print("CRITICAL: High memory usage! Consider reducing batch size or dataset size.")
            gc.collect()  # Force garbage collection
    return memory_mb

def main():
    parser = argparse.ArgumentParser(description='Run E-PURIFIER Comprehensive Evaluation')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'purchase100', 'facescrub530', 
                                'texas100', 'location', 'utkface', 'all'],
                        help='Dataset to evaluate (default: cifar10)')
    parser.add_argument('--epochs', type=int, default=35,
                        help='Number of training epochs (default: 35)')
    parser.add_argument('--comprehensive', action='store_true',
                        help='Enable comprehensive evaluation with all 9 attacks')
    parser.add_argument('--skip-train', action='store_true',
                        help='Load saved model checkpoint; retrain defense only')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test with reduced epochs for debugging')
    
    args = parser.parse_args()
    
    print("="*80)
    print("E-PURIFIER COMPREHENSIVE EVALUATION RUNNER")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Comprehensive Mode: {'ENABLED' if args.comprehensive else 'DISABLED'}")
    print(f"Quick Test: {'ENABLED' if args.quick_test else 'DISABLED'}")
    print("="*80)
    
    # Adjust epochs for quick test
    if args.quick_test:
        args.epochs = min(args.epochs, 10)
        print("Quick test mode: Using reduced epochs for faster execution")
    
    # Import and run the main evaluation
    try:
        from purifier import main_enhanced
        
        # Override sys.argv for the main function
        sys.argv = [
            'run_comprehensive_evaluation.py',
            '--dataset', args.dataset,
            '--epochs', str(args.epochs)
        ]
        
        if args.comprehensive:
            sys.argv.append('--comprehensive')
        if args.skip_train:
            sys.argv.append('--skip-train')
        
        print("\nStarting E-PURIFIER evaluation...")
        main_enhanced()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        if args.comprehensive:
            print("\nGenerated files:")
            print("- e_purifier_comprehensive_results.json")
            print("- *_comprehensive_results.json (per dataset)")
            print("- table_1_complete_results.csv/.tex")
            print("- table_2_purifier_comparison.csv/.tex")
            print("- table_3_inference_time.csv/.tex")
            print("- table_4_mi_bounds_detection.csv/.tex")
            print("- attack_breakdown_figure.png")
            print("- privacy_utility_tradeoff.png")
            print("- roc_curves.png")
            print("- comprehensive_evaluation_report.md")
        else:
            print("\nGenerated files:")
            print("- e_purifier_all_results.json")
            print("- results_summary.png")
            print("- *_training_curves.png")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all required files are present:")
        print("- purifier.py")
        print("- comprehensive_evaluation.py")
        print("- results_table_generator.py")
        return 1
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
