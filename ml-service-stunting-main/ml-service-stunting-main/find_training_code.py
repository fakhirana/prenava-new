import json

with open("stunting_pregnancy_final.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("extracted_cells.txt", "w", encoding="utf-8") as out:
    for idx, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))
        if any(keyword in source for keyword in ["stunting_model.pkl", "joblib.dump", "LGBMClassifier", "best_model", "save_model", "stunting_scaler.pkl"]):
            out.write(f"\n{'='*50}\nCELL {idx} ({cell.get('cell_type')})\n{'='*50}\n")
            out.write(source)
print("Extracted matching cells to extracted_cells.txt")
