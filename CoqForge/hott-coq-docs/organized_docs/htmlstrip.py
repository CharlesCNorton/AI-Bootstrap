import os
from bs4 import BeautifulSoup

def convert_html_to_plaintext(source_dir, output_file):
    """
    Recursively convert all HTML files in a directory (and subdirectories) to a single plaintext file.

    :param source_dir: Root directory containing HTML files.
    :param output_file: Path to the resulting plaintext file.
    """
    try:
        # Open the output file in write mode
        with open(output_file, "w", encoding="utf-8") as outfile:
            processed_files = 0

            # Walk through all subdirectories and files
            for root, dirs, files in os.walk(source_dir):
                for filename in files:
                    if filename.endswith(".html"):
                        file_path = os.path.join(root, filename)
                        try:
                            # Read and parse the HTML file
                            with open(file_path, "r", encoding="utf-8") as infile:
                                soup = BeautifulSoup(infile, "html.parser")
                                text = soup.get_text(strip=True)  # Extract text from HTML

                                if text.strip():  # Ensure non-empty content
                                    # Add a header for the file's content in the plaintext
                                    outfile.write(f"--- {os.path.relpath(file_path, source_dir)} ---\n")
                                    outfile.write(text + "\n\n")
                                    processed_files += 1
                                else:
                                    print(f"Warning: No readable content in {file_path}")
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
    output_file_path = r"C:\Users\cnort\Desktop\hott_docs\combined_plaintext.txt"  # Output file path

    # Convert all HTML files to a single plaintext file
    convert_html_to_plaintext(source_directory, output_file_path)
