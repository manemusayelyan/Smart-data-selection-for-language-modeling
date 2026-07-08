import os
import sys
import argparse
import subprocess

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--datasets', type=str, default='gsm8k,svamp,asdiv,mawps,tabmwp,mathqa,sat_math,mmlu_stem,math')
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args()
    
    harness_dir = '/nfs/dgx/home/undergrad/mane/llama_CT/math-evaluation-harness'
    if not os.path.exists(harness_dir):
        print(f"ERROR: {harness_dir} not found. Run installation script first.")
        sys.exit(1)
    
    os.chdir(harness_dir)
    print(f"Working directory: {os.getcwd()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Datasets: {args.datasets}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    cmd = [
        'python', 'math_eval.py',
        '--model_name_or_path', args.checkpoint,
        '--data_names', args.datasets,
        '--output_dir', args.output_dir,
        '--split', 'test',
        '--prompt_type', 'cot',
        '--temperature', '0.0',
        '--use_vllm',
        '--use_safetensors'
    ]
    
    print("Running:", ' '.join(cmd))
    print("="*80)
    subprocess.run(cmd)
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()