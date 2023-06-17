import pandas as pd


def generic_reader(path: str, **args) -> pd.DataFrame:
    """
    Generic function that handles reading parquet, csv, and json files
    returns a DataFrame.

    :param path : filepath
    :type path : string

    :return type : pd.DataFrame

    """
    
    format_type = path.split('.')[-1]
    try:
        func = getattr(pd, f'read_{format_type}')
        if format_type == 'parquet':
            engine = 'auto'
            return func(path, engine)

        if format_type == 'csv':
            if len(args) != 0:
                return func(path, index_col = args['index_col'])
            
            return func(path)
        
        if format_type == 'json':
            return func(path, lines=True)
    except Exception as err:
        raise TypeError("format not handled") from err
    
def generic_file_saver(data_frame : pd.DataFrame , file_type : str, file_path : str):
    """
    Generic function that handles writing files supported by pandas.

    :param data_frame : targe dataFrame
    :type data_frame : pd.DatraFrame
    

    :param file_type : file_type to save
    :type file_type : string

    :param file_path : filepath
    :type file_path : string

    """
    try:
        func = getattr(data_frame, f'to_{file_type}')

        func(file_path)
    except Exception as err:
        print(err)

if __name__ == "__main__":
    pass
