import os
import json
import argparse
import shutil
import re

def reverse_cell_modification(source):
    """
    Reverses the specific modifications made by the original patching script.

    Args:
        source (list or str): The source code from the cell.

    Returns:
        list or None: Returns the reverted list of source lines if changes were made,
                      otherwise returns None.
    """
    # Ensure the source is a single string for uniform processing.
    if isinstance(source, list):
        original_code = "".join(source)
    elif isinstance(source, str):
        original_code = source
    else:
        return None

    # --- Rule 1: Check if this cell was modified by the patch script ---
    # The presence of the unsloth import is our key indicator.
    if "from unsloth import is_bfloat16_supported" not in original_code:
        return None

    # --- Rule 2: Remove the specific lines added by the script ---
    lines = original_code.split('\n')
    
    reverted_lines = []
    was_changed = False
    # This flag helps remove a single blank line that might be left
    # after removing the import statement.
    import_line_removed = False

    for line in lines:
        # Strip leading/trailing whitespace to make matching robust
        stripped_line = line.strip()

        # Use regex to match the lines we want to remove.
        # This handles variations in spacing around the equals sign.
        is_bf16_line = re.match(r"bf16\s*=\s*is_bfloat16_supported\(\),?", stripped_line)
        is_fp16_line = re.match(r"fp16\s*=\s*not is_bfloat16_supported\(\),?", stripped_line)
        is_import_line = (stripped_line == "from unsloth import is_bfloat16_supported")

        if is_bf16_line or is_fp16_line:
            was_changed = True
            # Skip this line by not adding it to the reverted_lines list.
            continue
        elif is_import_line:
            was_changed = True
            import_line_removed = True
            # Skip the import line itself.
            continue
        elif import_line_removed and stripped_line == "":
            # If the import line was just removed, and this line is blank,
            # skip this blank line to prevent an empty gap at the top.
            # This handles the case where there was a blank line after the import.
            import_line_removed = False # Reset flag so we only do this once.
            continue
        else:
            reverted_lines.append(line)
            # Reset flag if we encounter a non-empty line
            if stripped_line != "":
                import_line_removed = False


    if not was_changed:
        return None

    # --- Final Step: Reconstruct cell source ---
    # Join the lines back together. The aggressive whitespace cleanup has been removed.
    final_code_str = "\n".join(reverted_lines)

    # The .ipynb format expects a list of strings, with each line ending in a newline.
    return final_code_str.splitlines(True)


def process_notebooks_in_directory(directory_path):
    """
    Scans a directory for .ipynb files and applies the reversal script.

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
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook_data = json.load(f)

                was_modified = False
                for cell in notebook_data.get('cells', []):
                    if cell.get('cell_type') == 'code':
                        reverted_source = reverse_cell_modification(cell.get('source', []))
                        if reverted_source:
                            cell['source'] = reverted_source
                            was_modified = True

                if was_modified:
                    print(f"  [+] Found and reverted a cell in '{filename}'.")
                    # 1. Create a new backup of the file before overwriting.
                    backup_path = notebook_path + ".bak"
                    shutil.copy2(notebook_path, backup_path)
                    
                    # 2. Write the reverted data back to the original file.
                    with open(notebook_path, 'w', encoding='utf-8') as f:
                        json.dump(notebook_data, f, indent=1, ensure_ascii=False)
                        f.write('\n')
                    print(f"  [+] Saved changes. Backup created at '{filename}.bak'.")
                else:
                    print(f"  [-] No changes needed for '{filename}'.")

            except Exception as e:
                print(f"  [!] An unexpected error occurred while processing {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Automate the REVERSAL of patching SFTTrainer cells in Jupyter Notebooks.""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='The path to the directory containing notebooks.\nDefaults to the current directory.'
    )
    
    args = parser.parse_args()
    TARGET_DIRECTORY = args.directory
    
    if not os.path.isdir(TARGET_DIRECTORY):
        print(f"Error: The specified directory '{TARGET_DIRECTORY}' does not exist.")
    else:
        process_notebooks_in_directory(TARGET_DIRECTORY)
        print("\nReversal script finished.")
