def combine_R3(folder_path):
    import pandas as pd
    import os
    from pathlib import Path

    current_path = Path.cwd()

    # Get a list of all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through the list of CSV files and read them into DataFrames
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, encoding = "ISO-8859-1")
        dfs.append(df)

    # Concatenate all DataFrames into one large DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    combined_df = combined_df.drop_duplicates(subset=['dc.title[en_US]'], keep='first')
    simplified_df = combined_df[['dc.title[en_US]', 'dc.description.abstract[en_US]', 'dc.identifier.uri[en_US]', 'dc.subject.keyword[en_US]', 'id', 'collection']]
    simplified_df.columns = ['title', 'abstract', 'uri', 'keyword', 'id', 'collection']

    # Optionally, save the combined DataFrame to a new CSV file
    files_combined = input('Please enter how would you like to name the combined metadata file, including the .csv\n')
    simplified_df.to_csv(files_combined, index=False)
    return files_combined

