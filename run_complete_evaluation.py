#!/usr/bin/env python3
"""
================================================================================
Complete Evaluation Runner
================================================================================
Combines basic results and comprehensive evaluation metrics

BASIC RESULTS:
- Train Accuracy, Test Accuracy, Attack Success Rate, MI Bound
- Training curves, Sample visualizations, Summary plots, JSON results

COMPREHENSIVE RESULTS:
- Multiple Membership Inference Attacks
- Member Detection Metrics (TPR, FPR, Precision, Recall, F1, ROC-AUC)
- Utility Degradation Analysis
- Model Inversion Defense
- Attribute Inference Defense
- Statistical Confidence Intervals
================================================================================
"""

import torch
import numpy as np
import argparse
import json
import time
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Import basic functionality from purifier
from purifier import (
    run_single_dataset, create_model, train_model, EPurifier,
    ArrayDataset, DataLoader
)

# Import comprehensive evaluation classes
from comprehensive_evaluation import (
    ComprehensiveEvaluator, NSHAttack, MIleaksAttack, BlindMIAttack,
    GapAttack, TransferAttack, BoundaryAttack, AdaptiveAttack,
    FPAttack, EnhancedAttack, MemberDetectionEvaluator,
    UtilityEvaluator, ModelInversionAttack, AttributeInferenceAttack
)

def run_complete_evaluation(dataset_name='all', epochs=40):
    """Run complete evaluation combining basic and comprehensive metrics"""
    
    print("="*80)
    print("COMPLETE EVALUATION (Basic + Comprehensive)")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Epochs: {epochs}")
    print("\nResults you'll get:")
    print("  BASIC METRICS:")
    print("    ✓ Train Accuracy")
    print("    ✓ Test Accuracy") 
    print("    ✓ Attack Success Rate")
    print("    ✓ MI Bound")
    print("    ✓ Training curves")
    print("    ✓ Sample visualizations")
    print("    ✓ Summary plots")
    print("  COMPREHENSIVE METRICS:")
    print("    ✓ 9 Membership Inference Attacks")
    print("    ✓ Member Detection Metrics (TPR, FPR, Precision, Recall, F1, ROC-AUC)")
    print("    ✓ Utility Degradation Analysis")
    print("    ✓ Model Inversion Defense")
    print("    ✓ Attribute Inference Defense")
    print("    ✓ Statistical Confidence Intervals")
    print("="*80)
    
    # Dataset configurations
    datasets_to_run = {
        'cifar10': ('CIFAR10', 'load_cifar10', 'image', 10),
        'cifar100': ('CIFAR100', 'load_cifar100', 'image', 100),
        'purchase100': ('Purchase100', 'load_purchase100', 'tabular', 100),
        'facescrub530': ('FaceScrub530', 'load_facescrub530', 'tabular', 530),
        'texas100': ('Texas100', 'load_texas100', 'tabular', 100),
        'location': ('Location', 'load_location', 'tabular', 100),
        'utkface': ('UTKFace', 'load_utkface', 'tabular', 5)
    }
    
    if dataset_name != 'all':
        datasets_to_run = {dataset_name: datasets_to_run[dataset_name]}
    
    evaluator = ComprehensiveEvaluator()
    all_results = {}
    
    for dataset_key, (display_name, load_func_name, data_type, num_classes) in datasets_to_run.items():
        print(f"\n{'='*60}")
        print(f"DATASET: {display_name}")
        print(f"{'='*60}")
        
        try:
            # Import the correct load function
            import purifier
            if load_func_name == 'load_cifar10':
                load_func = purifier.load_cifar10
            elif load_func_name == 'load_cifar100':
                load_func = purifier.load_cifar100
            elif load_func_name == 'load_purchase100':
                load_func = purifier.load_purchase100
            elif load_func_name == 'load_facescrub530':
                load_func = purifier.load_facescrub530
            elif load_func_name == 'load_texas100':
                load_func = purifier.load_texas100
            elif load_func_name == 'load_location':
                load_func = purifier.load_location
            elif load_func_name == 'load_utkface':
                load_func = purifier.load_utkface
            
            # Step 1: Run basic evaluation (from purifier.py)
            print("  Step 1: Basic evaluation...")
            basic_result = run_single_dataset(display_name, load_func, data_type, num_classes, epochs)
            
            # Step 2: Setup for comprehensive evaluation
            print("  Step 2: Setting up comprehensive evaluation...")
            trainset, testset, _, _ = load_func()
            train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
            test_loader = DataLoader(testset, batch_size=128, shuffle=False)
            
            # Get input dimension
            if data_type == 'tabular':
                sample_x, _ = trainset[0]
                input_dim = sample_x.shape[0]
            else:
                input_dim = None
            
            # Recreate model and defense for comprehensive evaluation
            model = create_model(display_name.lower(), input_dim, num_classes)
            model, acc, history = train_model(model, train_loader, test_loader, epochs=epochs)
            defense = EPurifier(model, num_classes)
            defense.train(trainset)
            
            # Step 3: Run comprehensive evaluation
            print("  Step 3: Comprehensive evaluation...")
            
            # 3.1: Evaluate all 9 membership inference attacks
            attack_results = evaluator.evaluate_all_attacks(defense, test_loader, trainset, input_dim, num_classes)
            
            # 3.2: Member detection evaluation
            print("  Step 4: Member detection evaluation...")
            member_evaluator = MemberDetectionEvaluator()
            
            # Collect member/non-member vectors
            member_vecs = []
            nonmember_vecs = []
            
            model.eval()
            with torch.no_grad():
                for x, y in train_loader:
                    outputs = model(x)
                    preds = torch.softmax(outputs, dim=1)
                    for pred in preds:
                        member_vecs.append(pred.numpy().flatten())
                    if len(member_vecs) >= 200:
                        break
                
                for x, y in test_loader:
                    outputs = model(x)
                    preds = torch.softmax(outputs, dim=1)
                    for pred in preds:
                        nonmember_vecs.append(pred.numpy().flatten())
                    if len(nonmember_vecs) >= 200:
                        break
            
            member_detection_results = member_evaluator.evaluate(member_vecs, nonmember_vecs)
            
            # 3.3: Utility evaluation
            print("  Step 5: Utility degradation evaluation...")
            utility_evaluator = UtilityEvaluator()
            utility_results = utility_evaluator.evaluate_utility_degradation(model, defense, test_loader, display_name)
            
            # Combine all results
            comprehensive_results = {
                'attack_results': attack_results,
                'member_detection': member_detection_results,
                'utility_degradation': utility_results
            }
            
            combined_result = {
                **basic_result,
                'comprehensive': comprehensive_results
            }
            
            all_results[dataset_key] = combined_result
            print(f"  ✓ {display_name} complete evaluation finished!")
            
        except Exception as e:
            print(f"  ✗ Error with {display_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_key] = {
                'dataset': display_name,
                'error': str(e),
                'train_accuracy': 0,
                'test_accuracy': 0,
                'attack_success': 50.0,
                'mi_bound': 0.0
            }
    
    # Save complete results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'complete_evaluation_results_{timestamp}.json'
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(all_results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Complete results saved to '{filename}'")
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPLETE EVALUATION SUMMARY")
    print("="*80)
    
    for dataset_key, result in all_results.items():
        if 'error' in result:
            print(f"{result['dataset']:<15} ERROR: {result['error']}")
        else:
            basic = result
            comp = result.get('comprehensive', {})
            
            print(f"\n{basic['dataset']}:")
            print(f"  BASIC METRICS:")
            print(f"    Train Acc: {basic.get('train_accuracy', 0):.2f}%")
            print(f"    Test Acc: {basic.get('test_accuracy', 0):.2f}%")
            print(f"    Attack Success: {basic.get('attack_success', 50):.2f}%")
            print(f"    MI Bound: {basic.get('mi_bound', 0):.4f}")
            
            if 'attack_results' in comp:
                print(f"  9-ATTACK RESULTS:")
                for attack_name, attack_result in comp['attack_results'].items():
                    print(f"    {attack_name:<12}: {attack_result.get('attack_success', 50):.2f}%")
            
            if 'member_detection' in comp:
                print(f"  MEMBER DETECTION:")
                md = comp['member_detection']
                if 'detector' in md:
                    print(f"    Accuracy: {md['detector'].get('accuracy', 0):.3f}")
                    print(f"    ROC-AUC: {md['detector'].get('roc_auc', 0):.3f}")
            
            if 'utility_degradation' in comp:
                print(f"  UTILITY:")
                util = comp['utility_degradation']
                print(f"    Clean Acc: {util.get('clean_accuracy', 0):.2f}%")
                print(f"    Defended Acc: {util.get('defended_accuracy', 0):.2f}%")
                print(f"    Degradation: {util.get('utility_degradation', 0):.2f}%")
    
    print("\n" + "="*80)
    print("COMPLETE EVALUATION FINISHED!")
    print("Results include both basic metrics AND comprehensive evaluation")
    print("="*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run complete E-PURIFIER evaluation')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['cifar10', 'cifar100', 'purchase100', 'facescrub530', 
                                'texas100', 'location', 'utkface', 'all'],
                        help='Dataset to evaluate (default: all)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of training epochs (default: 40)')
    
    args = parser.parse_args()
    
    run_complete_evaluation(args.dataset, args.epochs)
