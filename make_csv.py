import json
import csv

def extract_structured_data(json_filename):
    try:
        with open(json_filename, 'r') as json_file:
            metadata = json.load(json_file)

            structured_data = []

            for filename, details in metadata.items():
                record = {
                    'filename': filename,
                    'label': details.get('label', ''),
                    'split': details.get('split', ''),
                    'original': details.get('original', '')
                }

                structured_data.append(record)

            return structured_data

    except FileNotFoundError:
        print(f"Error: File '{json_filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_filename}'.")
        return None

def save_to_csv(structured_data, csv_filename):
    if structured_data:
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['filename', 'label', 'split', 'original']
                csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)

                csvwriter.writeheader()
                csvwriter.writerows(structured_data)

            print(f"CSV file '{csv_filename}' created successfully.")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No structured data to save.")

if __name__ == "__main__":
    # Replace 'your_metadata_file.json' and 'output_metadata.csv' with actual file names
    json_filename = 'C:/Users/vansh/Desktop/DEEP/dfdc_train_part_46/dfdc_train_part_46/metadata.json'
    csv_filename = 'C:/Users/vansh/Desktop/DEEP/metadata.csv'

    # Call the function to extract structured data
    structured_data = extract_structured_data(json_filename)

    # Call the function to save structured data to CSV
    save_to_csv(structured_data, csv_filename)
