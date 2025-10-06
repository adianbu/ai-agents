import tabula
import pandas as pd
import sys

def extract_tables_from_pdf(pdf_path="it.pdf"):
    """Extract tables from a PDF file using tabula-py."""
    try:
        # Read all tables from the PDF (not just page 1)
        tables = tabula.read_pdf(pdf_path, pages="all")
        
        if tables is None:
            print("No tables found in the PDF.")
            return
        
        # Handle case where only one table is returned (not in a list)
        if not isinstance(tables, list):
            tables = [tables]
        
        # Print the extracted table(s)
        for i, table in enumerate(tables):
            print(f"Table {i+1}:")
            print(table)
            print()

        # Save all tables to a single Excel file with each table on a separate sheet
        if tables:
            with pd.ExcelWriter("all_tables.xlsx") as writer:
                for i, table in enumerate(tables):
                    sheet_name = f"Table_{i+1}"
                    # Limit sheet name to 31 characters (Excel limit)
                    sheet_name = sheet_name[:31]
                    table.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"All {len(tables)} tables saved to 'all_tables.xlsx'")
            print("Each table is on a separate sheet named Table_1, Table_2, etc.")
            print("In Excel, click on the sheet tabs at the bottom to view each table.")
        else:
            print("No tables to save.")

    except FileNotFoundError:
        print(f"Error: PDF file '{pdf_path}' not found.")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"Error extracting tables: {str(e)}")

if __name__ == "__main__":
    # If a PDF file path is provided as a command line argument, use it
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        extract_tables_from_pdf(pdf_path)
    else:
        # Otherwise, use the default file name
        extract_tables_from_pdf()