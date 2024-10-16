import polars as pl
import json

def load_txt(file_path: str) -> list:
    """
    Load data from a text file and return a list of tuples.

    Args:
        file_path (str): The path to the text file.

    Returns:
        list: A list of tuples, where each tuple contains (prompt, response).

    The text file should have the following format like a csv file:
    Prompt1,Response1
    Prompt2,Response2
    ...
    """
    data_list = []
    with open(file_path, 'r') as file:
        data = json.load(file)
    for item in data:
        data_list.append((item['prompt'], item['response']))
    return data_list

def load_dataset(file_path: str) -> list:
    """
    Load data from a CSV, Excel, or JSON file and return a list of tuples.

    Args:
        file_path (str): The path to the CSV, Excel, or JSON file.

    Returns:
        list: A list of tuples, where each tuple contains (prompt, response).

    Raises:
        ValueError: If the file format is not supported.

    The JSON file should have the following format:
    [
        {"prompt": "Prompt1", "response": "Response1"},
        {"prompt": "Prompt2", "response": "Response2"},
        ...
    ]

    The CSV or Excel file should have the following format:
    Prompt1,Response1
    Prompt2,Response2
    ...
    """
    data_list = []
    if file_path.endswith('.csv'):
        df = pl.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pl.read_excel(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as file:
            data = json.load(file)
        for item in data:
            data_list.append((item['prompt'], item['response']))
        return data_list
    else:
        raise ValueError("Unsupported file format. Please provide a CSV, Excel, or JSON file.")
    
    for row in df.iter_rows(named=True):
        data_list.append((row['prompt'], row['response']))
    
    return data_list
