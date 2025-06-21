import os
import json
import argparse
import shutil

def modify_notebook_cell(source):
    """
    Modifies the source code of a notebook cell according to specific rules.

    Args:
        source (list or str): The source code from the cell, which can be a list of strings or a single string.

    Returns:
        list or None: Returns the modified list of source lines if changes were made, 
                      otherwise returns None.
    """
    # Ensure the source is a single string for uniform processing.
    # Notebooks can store source as a list of lines or a single string.
    if isinstance(source, list):
        original_code = "".join(source)
    elif isinstance(source, str):
        original_code = source
    else:
        # If the source format is unexpected, skip this cell.
        return None

    # --- Rule 1: Check if this is the correct cell to modify ---
    # It must contain 'SFTTrainer(' and 'SFTConfig(' to be specific.
    # It must NOT already contain the unsloth import, to prevent re-editing.
    if (not "SFTTrainer(" in original_code  
        or not "GRPOTrainer(" in original_code 
        or not "DPOTrainer(" in original_code ) \
       and "from unsloth import is_bfloat16_supported" in original_code:
        return None

    # --- Rule 2: Add the required import statement at the beginning ---
    modified_code_str = "from unsloth import is_bfloat16_supported\n\n" + original_code

    lines = modified_code_str.split('\n')
    
    # --- Rule 3: Remove any existing bf16 or fp16 parameters ---
    # This prevents duplicate parameters if the user had them in a different format.
    lines_without_old_params = [
        line for line in lines 
        if not (line.strip().startswith("bf16 =") or line.strip().startswith("fp16 =") or line.strip().startswith("bf16=") or line.strip().startswith("fp16="))
    ]

    # --- Rule 4: Insert new parameters after 'per_device_train_batch_size' ---
    final_lines = []
    was_inserted = False
    for line in lines_without_old_params:
        final_lines.append(line)
        if "per_device_train_batch_size" in line and not was_inserted:
            # Determine indentation from the anchor line for correct formatting.
            indentation = ' ' * (len(line) - len(line.lstrip(' ')))
            
            # check if the `per_device_train_batch_size` is using format `per_device_train_batch_size = 1` or `per_device_train_batch_size=1`
            if "per_device_train_batch_size =" in line:
                # Create the new lines with the correct indentation.
                bf16_line = f"{indentation}bf16 = is_bfloat16_supported(),"
                fp16_line = f"{indentation}fp16 = not is_bfloat16_supported(),"
            else:
                # If no space, we add a newline before the new parameters.
                bf16_line = f"{indentation}bf16=is_bfloat16_supported(),"
                fp16_line = f"{indentation}fp16=not is_bfloat16_supported(),"
            
            final_lines.extend([bf16_line, fp16_line])
            was_inserted = True
    
    # If the anchor line wasn't found, something is wrong, so don't change the cell.
    if not was_inserted:
        return None

    # --- Final Step: Reconstruct cell source for the notebook format ---
    final_code_str = "\n".join(final_lines)
    # The .ipynb format expects a list of strings, with each line ending in a newline.
    return final_code_str.splitlines(True)


def process_notebooks_in_directory(directory_path):
    """
    Scans a directory for .ipynb files and applies modifications safely.
    
    Args:
        directory_path (str): The path to the directory containing the notebooks.
    """
    print(f"Scanning directory: '{directory_path}'...")
    
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if not filename.endswith(".ipynb"):
                continue

            notebook_path = os.path.join(root, filename)
            print(f"--- Processing: {notebook_path} ---")
            
            try:
                # Read the entire notebook into memory first.
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook_data = json.load(f)

                was_modified = False
                # Iterate through cells to find candidates for modification.
                for cell in notebook_data.get('cells', []):
                    if cell.get('cell_type') == 'code':
                        modified_source = modify_notebook_cell(cell.get('source', []))
                        if modified_source:
                            cell['source'] = modified_source
                            was_modified = True

                # Only perform file operations if a change was actually made.
                if was_modified:
                    print(f"  [+] Found and modified a cell in '{filename}'.")
                    # 1. Create backup of the original file.
                    backup_path = notebook_path + ".bak"
                    shutil.copy2(notebook_path, backup_path)
                    
                    # 2. Write the modified data back to the original file path.
                    with open(notebook_path, 'w', encoding='utf-8') as f:
                        json.dump(notebook_data, f, indent=1, ensure_ascii=False)
                        f.write('\n')
                    print(f"  [+] Saved changes. Original is now '{filename}.bak'.")
                else:
                    print(f"  [-] No changes needed for '{filename}'.")

            except FileNotFoundError:
                print(f"  [!] Error: File not found. Skipping {notebook_path}")
            except json.JSONDecodeError:
                print(f"  [!] Error: Could not parse JSON. Skipping broken notebook: {notebook_path}")
            except Exception as e:
                print(f"  [!] An unexpected error occurred while processing {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Automate modifications to SFTTrainer cells in Jupyter Notebooks.
        Adds the 'unsloth' import and bf16/fp16 flags based on unsloth's support function.""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='The path to the directory containing notebooks.\nDefaults to the current directory if not provided.'
    )
    
    args = parser.parse_args()
    TARGET_DIRECTORY = args.directory
    
    if not os.path.isdir(TARGET_DIRECTORY):
        print(f"Error: The specified directory '{TARGET_DIRECTORY}' does not exist.")
    else:
        process_notebooks_in_directory(TARGET_DIRECTORY)
        print("\nScript finished.")
