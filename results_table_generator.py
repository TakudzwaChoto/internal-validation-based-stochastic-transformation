"""
================================================================================
Results Table Generator
================================================================================
Generates tables and figures from evaluation results
================================================================================
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


class ResultsTableGenerator:
    """Generate tables and figures from comprehensive evaluation results"""
    
    def __init__(self):
        self.results_file = 'e_purifier_comprehensive_results.json'
        self.output_dir = 'results_tables'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_results(self):
        """Load comprehensive results from JSON file"""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return []
    
    def generate_all_tables_and_figures(self):
        """Generate all tables and figures"""
        print("    Generating comprehensive results tables and figures...")
        results = self.load_results()
        
        if not results:
            print("    Warning: No results found to generate tables/figures")
            return
        
        # Generate tables
        self.generate_table_1_complete_results(results)
        self.generate_table_2_purifier_comparison(results)
        self.generate_table_3_inference_time(results)
        self.generate_table_4_mi_bounds_detection(results)
        
        # Generate figures
        self.generate_attack_breakdown_figure(results)
        self.generate_privacy_utility_tradeoff(results)
        self.generate_roc_curves(results)
        
        # Generate report
        self.generate_comprehensive_report(results)
        
        print("    ✓ All tables and figures generated successfully")
    
    def generate_table_1_complete_results(self, results):
        """Generate Table 1: Complete Results"""
        data = []
        for r in results:
            dataset = r['dataset']
            train_acc = r.get('train_accuracy', 0)
            test_acc = r.get('test_accuracy', 0)
            attack_success = r.get('attack_success', 0)
            mi_bound = r.get('mi_bound', 0)
            
            # Get comprehensive attack results if available
            comp_results = r.get('comprehensive_evaluation', {})
            attack_results = comp_results.get('attack_results', {})
            
            row = {
                'Dataset': dataset,
                'Train Acc (%)': f"{train_acc:.2f}",
                'Test Acc (%)': f"{test_acc:.2f}",
                'Attack Success (%)': f"{attack_success:.2f}",
                'MI Bound': f"{mi_bound:.4f}"
            }
            
            # Add individual attack results
            for attack_name, attack_data in attack_results.items():
                row[f"{attack_name} (%)"] = f"{attack_data.get('success_rate', 0):.2f}"
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, 'table_1_complete_results.csv')
        df.to_csv(csv_path, index=False)
        
        # Save as LaTeX
        tex_path = os.path.join(self.output_dir, 'table_1_complete_results.tex')
        with open(tex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        
        print(f"    ✓ Generated table_1_complete_results")
    
    def generate_table_2_purifier_comparison(self, results):
        """Generate Table 2: E-PURIFIER Comparison"""
        data = []
        for r in results:
            dataset = r['dataset']
            test_acc = r.get('test_accuracy', 0)
            attack_success = r.get('attack_success', 0)
            
            # Get member detection results
            comp_results = r.get('comprehensive_evaluation', {})
            member_detection = comp_results.get('member_detection', {})
            detection_acc = member_detection.get('accuracy', 0)
            
            # Get utility metrics
            utility = comp_results.get('utility_metrics', {})
            utility_acc = utility.get('accuracy', 0)
            
            row = {
                'Dataset': dataset,
                'Test Acc (%)': f"{test_acc:.2f}",
                'Attack Success (%)': f"{attack_success:.2f}",
                'Member Detection Acc (%)': f"{detection_acc:.2f}",
                'Utility Acc (%)': f"{utility_acc:.2f}"
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        csv_path = os.path.join(self.output_dir, 'table_2_purifier_comparison.csv')
        df.to_csv(csv_path, index=False)
        
        tex_path = os.path.join(self.output_dir, 'table_2_purifier_comparison.tex')
        with open(tex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        
        print(f"    ✓ Generated table_2_purifier_comparison")
    
    def generate_table_3_inference_time(self, results):
        """Generate Table 3: Inference Time"""
        data = []
        for r in results:
            dataset = r['dataset']
            train_time = r.get('train_time_min', 0)
            defense_time = r.get('defense_time_min', 0)
            total_time = train_time + defense_time
            
            row = {
                'Dataset': dataset,
                'Training Time (min)': f"{train_time:.2f}",
                'Defense Training Time (min)': f"{defense_time:.2f}",
                'Total Time (min)': f"{total_time:.2f}"
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        csv_path = os.path.join(self.output_dir, 'table_3_inference_time.csv')
        df.to_csv(csv_path, index=False)
        
        tex_path = os.path.join(self.output_dir, 'table_3_inference_time.tex')
        with open(tex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        
        print(f"    ✓ Generated table_3_inference_time")
    
    def generate_table_4_mi_bounds_detection(self, results):
        """Generate Table 4: MI Bounds and Detection"""
        data = []
        for r in results:
            dataset = r['dataset']
            mi_bound = r.get('mi_bound', 0)
            
            comp_results = r.get('comprehensive_evaluation', {})
            member_detection = comp_results.get('member_detection', {})
            
            row = {
                'Dataset': dataset,
                'MI Bound (nats)': f"{mi_bound:.4f}",
                'Detection Accuracy (%)': f"{member_detection.get('accuracy', 0):.2f}",
                'Detection Precision': f"{member_detection.get('precision', 0):.2f}",
                'Detection Recall': f"{member_detection.get('recall', 0):.2f}"
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        csv_path = os.path.join(self.output_dir, 'table_4_mi_bounds_detection.csv')
        df.to_csv(csv_path, index=False)
        
        tex_path = os.path.join(self.output_dir, 'table_4_mi_bounds_detection.tex')
        with open(tex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        
        print(f"    ✓ Generated table_4_mi_bounds_detection")
    
    def generate_attack_breakdown_figure(self, results):
        """Generate attack breakdown figure"""
        if not results:
            return
        
        datasets = [r['dataset'] for r in results]
        comp_results = [r.get('comprehensive_evaluation', {}).get('attack_results', {}) for r in results]
        
        # Get all attack names
        attack_names = []
        if comp_results and comp_results[0]:
            attack_names = list(comp_results[0].keys())
        
        if not attack_names:
            print("    Warning: No attack results found for breakdown figure")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(datasets))
        width = 0.08
        
        for i, attack_name in enumerate(attack_names):
            attack_success_rates = []
            for cr in comp_results:
                attack_data = cr.get(attack_name, {})
                attack_success_rates.append(attack_data.get('success_rate', 50))
            
            ax.bar(x + i * width, attack_success_rates, width, label=attack_name)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Attack Success Rate (%)')
        ax.set_title('Attack Success Rate Breakdown by Dataset')
        ax.set_xticks(x + width * len(attack_names) / 2)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Target (50%)')
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, 'attack_breakdown_figure.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Generated attack_breakdown_figure")
    
    def generate_privacy_utility_tradeoff(self, results):
        """Generate privacy-utility tradeoff figure"""
        if not results:
            return
        
        datasets = [r['dataset'] for r in results]
        test_accs = [r.get('test_accuracy', 0) for r in results]
        attack_success = [r.get('attack_success', 0) for r in results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(test_accs, attack_success, s=200, alpha=0.7)
        
        for i, dataset in enumerate(datasets):
            ax.annotate(dataset, (test_accs[i], attack_success[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Test Accuracy (%)')
        ax.set_ylabel('Attack Success Rate (%)')
        ax.set_title('Privacy-Utility Tradeoff')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Target (50%)')
        ax.axvline(x=95, color='orange', linestyle='--', alpha=0.5, label='Target (95%)')
        ax.legend()
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, 'privacy_utility_tradeoff.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Generated privacy_utility_tradeoff")
    
    def generate_roc_curves(self, results):
        """Generate ROC curves (simplified version)"""
        if not results:
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        # Plot ideal point
        ax.scatter([0], [1], s=100, c='green', marker='*', label='Ideal', zorder=5)
        
        # Plot dataset performance points
        for r in results:
            attack_success = r.get('attack_success', 50) / 100
            # Simplified: use attack success as TPR, assume FPR = 1 - TPR
            tpr = attack_success
            fpr = 1 - attack_success
            ax.scatter([fpr], [tpr], s=200, alpha=0.7, label=r['dataset'])
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve (Simplified)')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, 'roc_curves.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Generated roc_curves")
    
    def generate_comprehensive_report(self, results):
        """Generate comprehensive evaluation report in Markdown"""
        report_path = os.path.join(self.output_dir, 'comprehensive_evaluation_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# E-PURIFIER Comprehensive Evaluation Report\n\n")
            f.write("## Summary\n\n")
            f.write(f"Total datasets evaluated: {len(results)}\n\n")
            
            f.write("## Results by Dataset\n\n")
            for r in results:
                f.write(f"### {r['dataset']}\n\n")
                f.write(f"- **Train Accuracy**: {r.get('train_accuracy', 0):.2f}%\n")
                f.write(f"- **Test Accuracy**: {r.get('test_accuracy', 0):.2f}%\n")
                f.write(f"- **Attack Success**: {r.get('attack_success', 0):.2f}%\n")
                f.write(f"- **MI Bound**: {r.get('mi_bound', 0):.4f} nats\n")
                f.write(f"- **Training Time**: {r.get('train_time_min', 0):.2f} min\n")
                f.write(f"- **Defense Time**: {r.get('defense_time_min', 0):.2f} min\n\n")
                
                comp_results = r.get('comprehensive_evaluation', {})
                attack_results = comp_results.get('attack_results', {})
                
                if attack_results:
                    f.write("#### Attack Breakdown\n\n")
                    for attack_name, attack_data in attack_results.items():
                        f.write(f"- **{attack_name}**: {attack_data.get('success_rate', 0):.2f}%\n")
                    f.write("\n")
            
            f.write("## Tables\n\n")
            f.write("Generated tables:\n")
            f.write("- table_1_complete_results.csv/.tex\n")
            f.write("- table_2_purifier_comparison.csv/.tex\n")
            f.write("- table_3_inference_time.csv/.tex\n")
            f.write("- table_4_mi_bounds_detection.csv/.tex\n\n")
            
            f.write("## Figures\n\n")
            f.write("Generated figures:\n")
            f.write("- attack_breakdown_figure.png\n")
            f.write("- privacy_utility_tradeoff.png\n")
            f.write("- roc_curves.png\n")
        
        print(f"    ✓ Generated comprehensive_evaluation_report")
