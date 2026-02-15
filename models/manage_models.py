"""
Model Management CLI
Quick tool to view, compare, and load saved models
"""
import argparse
from model_persistence import ModelPersistence


def main():
    parser = argparse.ArgumentParser(description='Manage saved models')
    parser.add_argument('action', choices=['list', 'compare', 'best', 'info'],
                       help='Action to perform')
    parser.add_argument('--model', type=str, help='Model name (for info/best actions)')
    parser.add_argument('--metric', type=str, default='accuracy',
                       help='Metric to use for comparison (default: accuracy)')
    
    args = parser.parse_args()
    
    persistence = ModelPersistence()
    
    if args.action == 'list':
        if args.model:
            persistence.list_models(args.model)
        else:
            persistence.list_models()
    
    elif args.action == 'compare':
        persistence.compare_models(args.metric)
    
    elif args.action == 'best':
        if not args.model:
            print("Error: --model required for 'best' action")
            return
        persistence.get_best_model(args.model, args.metric)
    
    elif args.action == 'info':
        if not args.model:
            print("Error: --model required for 'info' action")
            return
        persistence.list_models(args.model)


if __name__ == '__main__':
    main()
