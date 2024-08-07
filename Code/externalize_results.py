import pandas as pd
import os

# purpose: store raw and processed output of network expansions.

def write_output_files(results: dict, edges: dict):
    for name, info in results.items():
        if not os.path.exists(f'Output/{name}'):
            os.mkdir(f'Output/{name}')
        for infotype, df in info.items():
            if type(df) in [pd.DataFrame, pd.Series]:
                df.to_csv(f'Output/{name}/{infotype}.csv.zip', compression='zip', encoding='utf-8')
    [df.to_csv(f'Output/{name}/edges.csv.zip', compression='zip', encoding='utf-8') for name, df in edges.items()]
    output_folders = set(os.listdir('Output')).intersection(results.keys())
    print('\n\nResult folder(s) in Output:\n', '\n'.join(output_folders), sep='')
    return(None)

