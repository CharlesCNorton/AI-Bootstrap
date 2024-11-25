import os
from bs4 import BeautifulSoup
import textwrap

def convert_html_to_plaintext(source_dir, output_file):
    """
    Recursively convert all HTML files in a directory (and subdirectories) to a single cleaned-up plaintext file.
    """
    try:
        # Open the output file in write mode
        with open(output_file, "w", encoding="utf-8") as outfile:
            processed_files = 0

            # Walk through all subdirectories and files
            for root, dirs, files in os.walk(source_dir):
                category = os.path.basename(root)  # Use folder name as category
                if files:
                    outfile.write(f"\n\n===== {category.upper()} =====\n\n")  # Add category header

                for filename in files:
                    if filename.endswith(".html"):
                        file_path = os.path.join(root, filename)
                        try:
                            # Read and parse the HTML file
                            with open(file_path, "r", encoding="utf-8") as infile:
                                soup = BeautifulSoup(infile, "html.parser")

                                # Remove irrelevant tags
                                for tag in soup(["script", "style", "meta", "nav"]):
                                    tag.decompose()

                                text = soup.get_text(separator="\n", strip=True)  # Extract text
                                clean_text = "\n".join(
                                    [line.strip() for line in text.splitlines() if line.strip()]
                                )  # Remove empty lines

                                # Skip files with insufficient content
                                if len(clean_text.split()) < 10:
                                    print(f"Skipping {file_path} due to insufficient content.")
                                    continue

                                # Add file-specific header and wrap text
                                outfile.write(f"\n--- {os.path.relpath(file_path, source_dir)} ---\n\n")
                                wrapped_text = "\n".join(
                                    textwrap.fill(line, width=80) for line in clean_text.splitlines()
                                )
                                outfile.write(wrapped_text + "\n\n")
                                processed_files += 1
                        except Exception as file_error:
                            print(f"Error processing file {file_path}: {file_error}")

            if processed_files == 0:
                print("No valid HTML files were processed.")
            else:
                print(f"Plaintext file successfully created: {output_file}")
    except Exception as e:
        print(f"Critical Error: {e}")


if __name__ == "__main__":
    # Specify the root directory and output file
    source_directory = r"C:\Users\cnort\Desktop\hott_docs\organized_docs"  # Update with your root directory
    output_file_path = r"C:\Users\cnort\Desktop\hott_docs\cleaned_plaintext.txt"  # Output file path

    # Convert all HTML files to a single cleaned-up plaintext file
    convert_html_to_plaintext(source_directory, output_file_path)
