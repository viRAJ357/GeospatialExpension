import json

with open("mlpaproject_final_.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("train_models.py", "w", encoding="utf-8") as out:
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            # Comment out shell commands like !pip install
            source_lines = source.split('\n')
            cleaned_lines = []
            for line in source_lines:
                if line.strip().startswith('!'):
                    cleaned_lines.append('# ' + line)
                elif 'google.colab' in line:
                    cleaned_lines.append('# ' + line)
                elif 'drive.mount' in line:
                    cleaned_lines.append('# ' + line)
                else:
                    line = line.replace("'/content/final_optimized_geospacial_dataset (1).csv'", "'Geospacial Data  .csv'")
                    line = line.replace('"/content/final_optimized_geospacial_dataset (1).csv"', '"Geospacial Data  .csv"')
                    line = line.replace('/content/drive/MyDrive/model and Preprocessing Objects MLPA project/', './')
                    line = line.replace('/content/drive/MyDrive/model and Preprocessing Objects MLPA project', './')
                    cleaned_lines.append(line)
            out.write('\n'.join(cleaned_lines) + "\n\n")

print("Successfully converted notebook to train_models.py")
