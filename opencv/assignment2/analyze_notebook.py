import json
import sys

def extract_code_cells(notebook_path):
    """Extract code cells from a Jupyter notebook"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        print(f"Successfully loaded notebook: {notebook_path}")
        print(f"Notebook format version: {notebook.get('nbformat', 'Unknown')}.{notebook.get('nbformat_minor', 'Unknown')}")
        
        code_cells = [cell for cell in notebook.get('cells', []) if cell.get('cell_type') == 'code']
        print(f"Found {len(code_cells)} code cells")
        
        for i, cell in enumerate(code_cells, 1):
            source = ''.join(cell.get('source', []))
            outputs = cell.get('outputs', [])
            execution_count = cell.get('execution_count')
            
            print(f"\n{'='*80}")
            print(f"CODE CELL {i} (Execution count: {execution_count})")
            print(f"{'-'*80}")
            print(source)
            
            if outputs:
                print(f"\n{'-'*80}")
                print(f"OUTPUTS:")
                print(f"{'-'*80}")
                for output in outputs:
                    output_type = output.get('output_type', '')
                    if output_type == 'stream':
                        print(f"Stream ({output.get('name', '')}):")
                        print(''.join(output.get('text', [])))
                    elif output_type == 'execute_result':
                        data = output.get('data', {})
                        if 'text/plain' in data:
                            print(f"Execute result:")
                            print(''.join(data['text/plain']))
                    elif output_type == 'error':
                        print(f"Error: {output.get('ename', '')}")
                        print(output.get('evalue', ''))
                        traceback = '\n'.join(output.get('traceback', []))
                        print(traceback)
                
        return True
    except Exception as e:
        print(f"Error processing notebook: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = "Assignment_2_Notebook.ipynb"
    extract_code_cells(notebook_path) 