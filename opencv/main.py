import argparse
import os
import sys
from q1 import run_orb_matching, compare_parameters
from q1_failure_analysis import analyze_image_pair, try_improve_matching, test_multiple_categories, find_failure_cases

def main():
    parser = argparse.ArgumentParser(description='Computer Vision Assignment 2')
    parser.add_argument('--question', type=str, choices=['q1', 'q2', 'q3'], default='q1',
                        help='Which question to run (default: q1)')
    parser.add_argument('--action', type=str, default='basic',
                        help='Which action to perform (default: basic)')
    parser.add_argument('--ref', type=str, default=None,
                        help='Path to reference image')
    parser.add_argument('--query', type=str, default=None,
                        help='Path to query image')
    parser.add_argument('--category', type=str, 
                        choices=['book_covers', 'museum_paintings', 'landmarks'], 
                        default='book_covers',
                        help='Image category to use (default: book_covers)')
    parser.add_argument('--id', type=str, default='001',
                        help='Image ID to use (default: 001)')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists('A2_smvs'):
        print("Error: A2_smvs dataset directory not found!")
        print("Please make sure the dataset is extracted to the correct location.")
        sys.exit(1)
    
    # Default paths based on category and ID
    if args.ref is None:
        args.ref = f"A2_smvs/{args.category}/Reference/{args.id}.jpg"
    if args.query is None:
        args.query = f"A2_smvs/{args.category}/Query/{args.id}.jpg"
    
    # Check if specific files exist
    if not os.path.exists(args.ref):
        print(f"Error: Reference image not found: {args.ref}")
        sys.exit(1)
    if not os.path.exists(args.query):
        print(f"Error: Query image not found: {args.query}")
        sys.exit(1)
    
    # Run the appropriate action based on arguments
    if args.question == 'q1':
        run_q1(args)
    elif args.question == 'q2':
        print("Question 2 not implemented in this script yet.")
    elif args.question == 'q3':
        print("Question 3 not implemented in this script yet.")

def run_q1(args):
    """Run Question 1 actions"""
    if args.action == 'basic':
        print(f"Running basic ORB matching on {args.ref} and {args.query}")
        results = run_orb_matching(args.ref, args.query)
        print(f"Found {results['num_matches']} good matches and {results['inlier_count']} inliers")
        
    elif args.action == 'analyze':
        print(f"Running detailed analysis on {args.ref} and {args.query}")
        analyze_image_pair(
            args.ref, args.query,
            f"Analysis of {args.category}/{args.id}",
            "Detailed analysis of ORB matching results"
        )
        
    elif args.action == 'compare':
        print(f"Comparing parameter settings for {args.ref} and {args.query}")
        results = compare_parameters(args.ref, args.query)
        
        # Print summary
        print("\nSummary of comparison:")
        for key, result in results.items():
            print(f"{key}: {result['num_matches']} good matches, {result['inlier_count']} inliers")
            
    elif args.action == 'find_failures':
        print("Finding success and failure cases across all categories")
        success_cases, failure_cases = find_failure_cases()
        
    elif args.action == 'test_categories':
        print("Testing images from multiple categories")
        results = test_multiple_categories()
        
    else:
        print(f"Unknown action: {args.action}")
        print("Available actions for q1: basic, analyze, compare, find_failures, test_categories")

if __name__ == "__main__":
    main() 