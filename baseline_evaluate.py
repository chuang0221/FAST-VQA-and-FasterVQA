import subprocess
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import os
import argparse
from tqdm import tqdm

import warnings
import sys

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class WarningSuppress:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

def run_vqa_prediction(model_type, video_path):
    """Run VQA prediction using specified model"""
    cmd = f"python vqa.py -m {model_type} -v {video_path}"
    try:
        output = subprocess.check_output(cmd, shell=True, universal_newlines=True,
                                         stderr=subprocess.DEVNULL)
        score_line = [line for line in output.split('\n') if "quality score" in line][0]
        score = float(score_line.split()[-1].strip('.'))
        return score
    except:
        print(f"Error processing video: {video_path}")
        return None

def evaluate_vqa_model(model_type, videos_dir, mos_file):
    """Evaluate VQA model against MOS scores"""
    sheet_name = None
    if "hdr2sdr" in videos_dir.lower():
        sheet_name = "HDR2SDR"
    elif "sdr" in videos_dir.lower():
        sheet_name = "SDR"
    
    if sheet_name is None:
        raise ValueError("Could not determine sheet_name from videos_dir. Path should contain 'hdr2sdr' or 'sdr'")

    mos_df = pd.read_excel(mos_file, sheet_name=sheet_name)
    
    results = []
    
    for _, row in tqdm(mos_df.iterrows(), total=len(mos_df), desc=f"Processing {sheet_name} videos"):
        video_name = row['vid']
        mos_score = row['mos']
        
        video_path = os.path.join(videos_dir, video_name)
        video_path = video_path + '.mp4'
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
            
        pred_score = run_vqa_prediction(model_type, video_path)
        if pred_score is not None:
            results.append({
                'video_name': video_name,
                'mos': mos_score,
                'predicted_score': pred_score * 5
            })

    results_df = pd.DataFrame(results)

    srocc = spearmanr(results_df['mos'], results_df['predicted_score'])[0]
    plcc = pearsonr(results_df['mos'], results_df['predicted_score'])[0]

    timestamp = pd.Timestamp.now().strftime('%Y%m%d')
    results_dir = os.path.join(os.path.dirname(__file__), 
                              'baseline_experiment', 
                              f'short-{sheet_name.lower()}',
                              model_type,
                              timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nResults for {model_type} on {sheet_name} sheet:")
    print(f"SROCC: {srocc:.4f}")
    print(f"PLCC: {plcc:.4f}")
    
    results_df.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
    
    with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
        f.write(f"SROCC: {srocc:.4f}\n")
        f.write(f"PLCC: {plcc:.4f}\n\n")
        
        results_df['abs_diff'] = abs(results_df['predicted_score'] - results_df['mos'])
        best_5 = results_df.nsmallest(5, 'abs_diff')
        worst_5 = results_df.nlargest(5, 'abs_diff')
        
        f.write("Top 5 Best Predictions:\n")
        f.write(best_5[['video_name', 'mos', 'predicted_score', 'abs_diff']].to_string())
        f.write("\n\nTop 5 Worst Predictions:\n")
        f.write(worst_5[['video_name', 'mos', 'predicted_score', 'abs_diff']].to_string())
    
    return srocc, plcc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='FAST-VQA',
                       help='VQA model type (e.g., FAST-VQA, FasterVQA)')
    parser.add_argument('--videos_dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('--mos_file', type=str, required=True,
                       help='Path to Excel file containing MOS scores')
    
    args = parser.parse_args()
    
    evaluate_vqa_model(args.model, args.videos_dir, args.mos_file)

if __name__ == "__main__":
    main()