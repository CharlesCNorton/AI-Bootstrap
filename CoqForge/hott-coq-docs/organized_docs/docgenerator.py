import os
import subprocess
import shutil
import glob


def generate_hott_docs(coq_hott_path, output_path):
    """
    Generate Unicode plaintext documentation for the Coq HoTT library using coqdoc.
    """
    try:
        # Check if coqdoc is installed
        result = subprocess.run(["coqdoc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("Error: coqdoc is not installed or not in your PATH.")
            return

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Collect all .v files from the coq_hott_path
        v_files = glob.glob(os.path.join(coq_hott_path, "**", "*.v"), recursive=True)
        if not v_files:
            print(f"No .v files found in {coq_hott_path}.")
            return

        # Generate documentation in Unicode plaintext
        coqdoc_command = [
            "coqdoc",
            "--utf8",            # Enable Unicode output
            "--plain",           # Output plaintext
            "--interpolate",     # Enhance formatting
            "--parse-comments",  # Include comments
            "-d", output_path    # Output directory
        ] + v_files

        print("Generating plaintext documentation... This may take some time.")
        result = subprocess.run(coqdoc_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode == 0:
            print(f"Documentation successfully generated in: {output_path}")
        else:
            print("Error while generating documentation:")
            print(result.stderr.decode("utf-8"))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def organize_docs_by_categories(source_dir):
    """
    Organize Unicode plaintext documentation into categories based on original structure.
    """
    try:
        # Create a folder for organized documentation
        organized_dir = os.path.join(source_dir, "organized_docs")
        os.makedirs(organized_dir, exist_ok=True)

        # Traverse and sort files based on their original folders
        for filename in os.listdir(source_dir):
            if filename.endswith(".txt"):
                # Parse the file name to infer its category
                category_name = filename.split(".")[0]  # Assuming filenames like Algebra.txt
                subfolder = find_category(category_name)

                if subfolder:
                    # Create subdirectory for category if it doesn't exist
                    category_dir = os.path.join(organized_dir, subfolder)
                    os.makedirs(category_dir, exist_ok=True)

                    # Move file to its category folder
                    source_file = os.path.join(source_dir, filename)
                    destination_file = os.path.join(category_dir, filename)
                    shutil.move(source_file, destination_file)

        print(f"Documentation successfully organized in: {organized_dir}")

    except Exception as e:
        print(f"Error organizing documentation: {e}")


def find_category(file_name):
    """
    Define a mapping for categories based on file names.
    """
    categories = {
        "Algebra": ["AbelianGroup", "Abelianization", "Algebra", "Rings", "Groups"],
        "Basics": ["Basics", "Utf8", "Overture", "Tactics"],
        "Categories": ["Category", "Functor", "Limits", "Grothendieck"],
        "Homotopy": ["PathGroupoids", "Suspension", "WhiteheadsPrinciple"],
        "HIT": ["Interval", "Quotient", "SetCone"],
        "Truncations": ["Trunc", "Truncations", "TruncType"],
    }

    for category, files in categories.items():
        if file_name in files:
            return category
    return "Miscellaneous"  # Default category for uncategorized files


if __name__ == "__main__":
    # Paths for HoTT theories and output directory
    coq_hott_path = r"C:\Users\cnort\HoTT\theories"  # Update this path to your HoTT theories
    output_path = r"C:\Users\cnort\Desktop\hott_docs"  # Update this path for output

    # Step 1: Generate Unicode plaintext documentation
    generate_hott_docs(coq_hott_path, output_path)

    # Step 2: Organize the generated files into categories
    organize_docs_by_categories(output_path)
