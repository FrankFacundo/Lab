import json
import os
import sys

from PyPDF2 import PdfReader


def count_pdf_pages(pdf_path):
    """Return the number of pages in the PDF at pdf_path."""
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return 0


def process_path(path):
    """
    Process a given path (directory or file). If it's a directory, recursively process
    its children and sum up their PDF page counts. If it's a PDF file, count its pages.
    Returns a node dict with keys:
        - name: basename of the file/directory
        - pages: total number of pages (aggregated)
        - children: list of child nodes (None for PDF files)
    """
    if os.path.isdir(path):
        # For directories, create a node with an empty children list.
        node = {"name": os.path.basename(path) or path, "pages": 0, "children": []}
        try:
            entries = sorted(os.listdir(path))
        except Exception as e:
            print(f"Error listing directory {path}: {e}")
            entries = []
        for entry in entries:
            full_path = os.path.join(path, entry)
            child_node = process_path(full_path)
            if child_node is not None:
                node["children"].append(child_node)
                node["pages"] += child_node["pages"]
        return node
    else:
        # Process only PDF files; ignore others.
        if path.lower().endswith(".pdf"):
            pages = count_pdf_pages(path)
            return {"name": os.path.basename(path), "pages": pages, "children": None}
        else:
            return None


def print_tree(node, prefix="", is_last=True):
    """
    Pretty-print the tree using branch-like markers.
    'prefix' holds the string to print before the node, and is_last determines
    whether the node is the last child (to choose the appropriate branch marker).
    """
    if node is None:
        return

    connector = "└── " if is_last else "├── "
    print(prefix + connector + f"{node['name']} ({node['pages']} pages)")

    if node["children"]:
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node["children"]):
            is_last_child = i == len(node["children"]) - 1
            print_tree(child, new_prefix, is_last_child)


def tree_to_json(node):
    """
    Convert the tree node into a JSON-friendly dictionary.
    Each node will have:
      - "text": a label combining the node's name and page count.
      - "state": a dict where "opened" can be set to True to have the node expanded by default.
      - "children": a list of child nodes (empty list for leaf nodes).
    """
    json_node = {
        "text": f"{node['name']} ({node['pages']} pages)",
        "state": {
            "opened": True
        },  # Set to False if you want nodes collapsed by default.
    }
    if node["children"]:
        json_node["children"] = [tree_to_json(child) for child in node["children"]]
    else:
        json_node["children"] = []
    return json_node


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)

    start_directory = sys.argv[1]
    tree = process_path(start_directory)

    # Print the tree in a pretty way.
    print(f"{tree['name']} ({tree['pages']} pages)")
    if tree["children"]:
        for i, child in enumerate(tree["children"]):
            is_last_child = i == len(tree["children"]) - 1
            print_tree(child, "", is_last_child)

    # Convert the tree to a JSON-friendly format.
    json_tree = tree_to_json(tree)

    # Export the JSON tree to a file.
    output_file = "tree.json"
    try:
        with open(output_file, "w") as f:
            json.dump(json_tree, f, indent=4, ensure_ascii=False)
        print(f"\nTree exported to JSON file: {output_file}")
    except Exception as e:
        print(f"Error writing JSON file: {e}")
