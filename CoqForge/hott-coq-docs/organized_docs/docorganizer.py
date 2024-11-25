import os
import shutil

def organize_html_docs(source_dir):
    """
    Organize HTML documentation into categories based on original structure.
    :param source_dir: Path to the directory containing generated HTML files.
    """
    try:
        # Create a folder for organized documentation
        organized_dir = os.path.join(source_dir, "organized_docs")
        os.makedirs(organized_dir, exist_ok=True)

        # Traverse and sort files based on their original folders
        for filename in os.listdir(source_dir):
            if filename.endswith(".html"):
                # Parse the file name to infer its category
                category_name = filename.split(".")[0]  # Assuming filenames like Algebra.html
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
    This is a simplified example. You can expand it based on your folder hierarchy.
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
    # Specify the source directory containing the generated HTML files
    source_directory = r"C:\Users\cnort\Desktop\hott_docs"
    organize_html_docs(source_directory)
