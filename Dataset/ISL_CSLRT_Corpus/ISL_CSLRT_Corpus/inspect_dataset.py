import os
import openpyxl

folder = r".\corpus_csv_files"

for filename in os.listdir(folder):

    if not filename.endswith(".xlsx"):
        continue

    path = os.path.join(folder, filename)

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)

    for sheet_name in workbook.sheetnames:

        sheet = workbook[sheet_name]

        print(f"\nSheet: {sheet_name}")
        print(f"Rows: {sheet.max_row}, Columns: {sheet.max_column}")

        for row in sheet.iter_rows(max_row=5, values_only=True):
            print(row)

    workbook.close()