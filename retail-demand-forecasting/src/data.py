from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

@dataclass(frozen=True)
class WalmartPaths():
    ''' Convenience store for Walmart paths '''
    data_dir: Path

    @property
    def store_path(self) -> Path:
        return self.data_dir / 'stores.csv'

    @property
    def features_path(self) -> Path:
        return self.data_dir / 'features.csv'
    
    @property
    def sales_path(self) -> Path:
        return self.data_dir / 'sales.csv'
    
REQUIRED_STORES_COLUMNS = { 'Store', 'Type', 'Size' }
REQUIRED_FEATURES_COLUMNS = { 'Store', 'Date', 'Temperature', 'Fuel_Price', 'MarkDown1', 
                             'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 
                             'Unemployment', 'IsHoliday'}
REQUIRED_SALES_COLUMNS = { 'Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Temperature', 
                          'Fuel_Price', 'CPI', 'Unemployment' }

def load_stores(path: str | Path) -> pd.DataFrame:
    '''
    Loads the Walmart stores CSV file

    Expected columns:
    Store, Type, Size
    '''
    path = Path(path)
    df = pd.read_csv(path)

    missing = REQUIRED_STORES_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(f'The `stores.csv` file is missing the following required columns: {sorted(missing)}')
                         
    if df.empty:
        raise ValueError('The "sales" dataframe is empty')
    
    return df

def load_features(path: str | Path) -> pd.DataFrame:
    '''
    Loads the Walmart features CSV dile

    Expected colums:
    Store, Date, Temperature, Fuel_Price, MarkDown1, MarkDown2, MarkDown3, MarkDown4,
    MarkDown5, CPI, Unemployment, IsHoliday
    '''
    path = Path(path)
    df = pd.read_csv(path, parse_dates = ['Date'])

    missing = REQUIRED_FEATURES_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(f'The "features.csv" file is missing the following required columns: {sorted(missing)}')
    
    if df.empty:
        raise ValueError('The "features" dataframe is empty')
    
    # Convert the 'Date' column to the `DateTime` data type
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    # Map 'Yes" and 'No' to '1' and '0', respectively
    df['IsHoliday'] = df['IsHoliday'].map({ True: 1, False: 0 })
    
    return df

def load_sales(path: str | Path) -> pd.DataFrame:
    '''
    Loads the Walmart sales CSV file
    
    Expected columns: 
    Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment
    '''
    path = Path(path)
    df = pd.read_csv(path)

    missing = REQUIRED_SALES_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(f'The "sales.csv" file is missing the following required columns: {sorted(missing)}')
    
    if df.empty:
        raise ValueError('The sales dataframe is empty')
    
    # Convert the 'Date' column to the `DateTime` data type
    df['Date'] = pd.to_datetime(df['Date'], format = '%d-%m-%Y')

    # Rename the 'Holiday_Flag' column to 'IsHoliday' to match the same column in the features file
    df = df.rename(columns = {'Holiday_Flag': 'IsHoliday'})

    # Drop the `Temperature`, `Fuel_Price`, `CPI`, and `Unemployment` columns since they are
    # already present in the `features` dataset
    extra_columns = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    df = df.drop(columns = extra_columns)

    return df


def load_walmart_data(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Loads (stores, features, sales) from a folder that contains the respective CSV files
    '''
    paths = WalmartPaths(Path(data_dir))

    if not paths.store_path.exists():
        raise FileNotFoundError(f'Missing file: {paths.store_path}')
    
    if not paths.features_path.exists():
        raise FileNotFoundError(f'Missing fle: {paths.features_path}')
    
    if not paths.sales_path.exists():
        raise FileNotFoundError(f'Missing file: {paths.sales_path}')
    
    stores = load_stores(paths.store_path)
    features = load_features(paths.features_path)
    sales = load_sales(paths.sales_path)

    return stores, features, sales

