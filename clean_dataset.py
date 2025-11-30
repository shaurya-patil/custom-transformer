import os
import glob

def clean_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    files = glob.glob(os.path.join(input_dir, "*.txt"))
    print(f"Found {len(files)} files to process.")

    total_pairs = 0
    
    for file_path in files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        
        unique_pairs = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    # Extract English and Target (first two columns)
                    english = parts[0].strip()
                    target = parts[1].strip()
                    
                    if english and target:
                        unique_pairs.add((english, target))
            
            # Extract language code from filename (e.g., 'afr.txt' -> 'afr')
            lang_code = os.path.splitext(filename)[0]
            
            # Write cleaned pairs with language code
            with open(output_path, 'w', encoding='utf-8') as f:
                for eng, tgt in unique_pairs:
                    # Format: English \t Target \t LangCode
                    f.write(f"{eng}\t{tgt}\t{lang_code}\n")
            
            print(f"Processed {filename}: {len(unique_pairs)} pairs saved.")
            total_pairs += len(unique_pairs)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Cleaning complete. Total pairs: {total_pairs}")
    print(f"Cleaned files saved to: {output_dir}")

if __name__ == "__main__":
    INPUT_DIR = r"c:\\Users\\shaur\\OneDrive\\Desktop\\DL_Projects\\CustomTransformer\\bilingual-sentence-pairs\\versions\\3"
    OUTPUT_DIR = r"c:\\Users\\shaur\\OneDrive\\Desktop\\DL_Projects\\CustomTransformer\\cleaned_data"
    
    clean_dataset(INPUT_DIR, OUTPUT_DIR)
