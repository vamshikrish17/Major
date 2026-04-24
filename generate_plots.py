import matplotlib.pyplot as plt
import numpy as np

# Set global highly readable professional styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif'})

def generate_inference_plot():
    # Data simulation based on sequential loop O(N) vs Batched tensor execution
    objects = np.array([1, 2, 5, 8, 10, 15, 20])
    
    # Old Sequential Architecture (Time scales linearly per extra predicted mask block)
    seq_time = objects * 85 + 25  # roughly 85ms per SAM sequential predict + base YOLO time
    
    # New Concurrent Tensor Batching (Only a slight overhead increase for batched matrix size)
    batched_time = 120 + objects * 5.5 # Flatter scaling curve due to parallel computation
    
    plt.figure(figsize=(8, 6))
    plt.plot(objects, seq_time, marker='o', color='#E63946', linewidth=2.5, label='Legacy Sequential Processing')
    plt.plot(objects, batched_time, marker='s', color='#1D3557', linewidth=2.5, label='Proposed Concurrent Tensor Batching')
    
    plt.title('Inference Latency vs. Detected Objects in Frame', fontweight='bold', pad=15)
    plt.xlabel('Number of Detected Objects (Instances)')
    plt.ylabel('Total Inference Time (ms)')
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('performance_speed.png', dpi=300)
    plt.close()

def generate_accuracy_plot():
    # Data simulation based on IoU improvement using Point + Box (Hybrid Prompting) vs Box Only
    categories = ['Generic Box Prompt', 'Proposed Hybrid Semantic Prompt']
    accuracy = [76.5, 94.2] # IoU Metric Percentages
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, accuracy, color=['#457B9D', '#2A9D8F'], width=0.5)
    
    plt.title('Segmentation Accuracy (Mean IoU) Comparison', fontweight='bold', pad=15)
    plt.ylabel('Intersection over Union (%)')
    plt.ylim(0, 100)
    
    # Add numerical labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval}%", ha='center', va='bottom', fontweight='bold')
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('mask_accuracy.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_inference_plot()
    generate_accuracy_plot()
    print("Successfully generated performance_speed.png and mask_accuracy.png")
