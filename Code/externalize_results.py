import os

import pandas as pd


# purpose: store raw and processed output of network expansions.

def write_output_files(results: dict):
    for name, info in results.items():
        if not os.path.exists(f'Output/{name}'):
            os.mkdir(f'Output/{name}')
        for infotype, df in info.items():
            if type(df) == pd.DataFrame:
                output_df = df.copy(deep=True).fillna('')
                if infotype == 'participants':
                    output_df.drop(index=0, inplace=True)
                sets = output_df.convert_dtypes().select_dtypes(set).applymap(lambda x: ' '.join([str(s) for s in x]), na_action='ignore')
                output_df.loc[:, sets.columns] = sets.copy()
                output_df.to_csv(f'Output/{name}/{infotype}.csv.zip', compression='zip', encoding='utf-8')
            elif type(df) in [list, set]:
                with open(f'Output/{name}/{infotype}.txt', 'w') as f:
                    f.write(' '.join(df))
            elif type(df) == pd.Series:
                df.to_csv(f'Output/{name}/{infotype}.csv.zip', compression='zip', encoding='utf-8')
    output_folders = set(os.listdir('Output')).intersection(results.keys())
    print('\n\nResult folder(s) in Output:\n', '\n'.join(output_folders), sep='')
    return None


# purpose: convert list-like entries in a dataframe to strings for easy read/write

def make_lists_printable(df):
   list_like_cols = df.loc[:, df.iloc[0].apply(type).apply(lambda x: x in [set, list])]
   printable_cols = list_like_cols.applymap(' '.join)
   printable_df = df.copy(deep=True)
   printable_df.loc[:, printable_cols.columns] = printable_cols
   return(printable_df)
