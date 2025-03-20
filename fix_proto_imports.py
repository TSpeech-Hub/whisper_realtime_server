"""
Script used in Makefile to fix the imports of the generated protobuf files.
"""
import os
import re

GEN_DIR = "src/generated"

for filename in os.listdir(GEN_DIR):
    # search for any python files in the generated directory
    if filename.endswith(".py"):
        filepath = os.path.join(GEN_DIR, filename)
        with open(filepath, "r") as f:
            content = f.read()
        
        # Adjust the import of the generated protobuf files to match py modules
        # this regex searches for lines that start with "import <proto_file>_pb2" and adjust to "from src.generated import <proto_file>_pb2"
        # this is necessary because the generated files is a submodule of the src package
        content = re.sub(r"^import (\w+_pb2)", r"from src.generated import \1", content, flags=re.MULTILINE)

        with open(filepath, "w") as f:
            f.write(content)

print("Proto imports fixed!")
