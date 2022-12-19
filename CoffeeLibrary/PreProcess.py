import pandas as pd
import re
from dataprep.clean import clean_country

class PreProcess:

    def __init__(self, path):
        # read csv into dataframe
        self.df = pd.read_csv(path, sep = ',', thousands=',')

    def clean_origin(self):
        # extract country name from string in 'origin'
        self.df = clean_country(self.df, 'origin')
        self.df['origin'] = self.df['origin_clean']
        self.df = self.df.drop(columns = 'origin_clean')  
        
            
    def clean_location(self):
        # extract state (for US) or country in string in 'location'
        self.df['location'] = self.df['location'].apply(lambda row: row.split(',')[-1])

    def standardize_price(self):
        # clean and standardize prices in 'est_price'
        # prices are in different currencies and are a ratio of different measurements (i.e. grams, ounces)
        
        # create 'price' column
        self.df['price'] = self.df['est_price'].apply(lambda row: str(row).split('/')[0].replace(',','')).apply(lambda row: re.findall(r'[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', str(row)))
        # drop rows with empty cells and covert price to float
        self.df['price'] = self.df['price'].apply(lambda row: row[0] if len(row) > 0 else None).astype(float)
        
        # create 'currency' column
        self.df['currency'] = self.df['est_price'].apply(lambda row: str(row).split('/')[0]).apply(lambda row: re.split(r'([\d.,]+)$', str(row),1)[0])
        # filter rows with ['$','NT $', 'NT$'] in currency column
        self.df = self.df.apply(lambda row: row[self.df['currency'].isin(['$','NT $', 'NT$'])])
        # convert all prices to USD
        for x in ['NT $', 'NT$']: self.df.loc[self.df['currency'] == x, 'price'] = self.df['price'] * 0.032 
        
        # split 'amount' and 'units' into separate columns (e.g. '8' + 'grams')
        self.df[['amount', 'units']] = self.df['est_price'].apply(lambda row: str(row).split('/')[-1].replace(',','')).str.split(' ', 1, expand = True)
        # filter rows with ['ounces','grams'] in the units column
        self.df = self.df.apply(lambda row: row[self.df['units'].isin(['ounces','grams'])])
        # convert all units to grams
        self.df.loc[self.df['units'] == 'ounces', 'price'] = self.df['price'] * 28.3495 # exchange rate as of 18/12/2022
        
        # drop nas for columns created
        self.df = self.df.dropna(axis = 0, subset = ['currency', 'price', 'amount', 'units'])
        # standardize price
        self.df['price'] = self.df['price'] / self.df['amount'].astype(float)

        # delete all columns created except 'price'
        self.df = self.df.drop(columns = ['currency', 'amount', 'units', 'est_price'])    

    def standardize_agtron(self):
        # clean and standardize agtron score
        # degree of roast can be measured with some precision through the use of an Agtron
        # whole-bean M-Basic reading: whole beans before grinding
        # ground reading: the same beans after grinding 
        
        # separate agtron reading into whole-bean M-Basic reading and ground reading
        self.df[['m_basic', 'ground_reading']] = self.df['agtron'].str.split('/', expand = True).astype(int)
        # drop 'agtron' column
        self.df = self.df.drop(columns = 'agtron')  
        
    def drop_cols(self):
        # drop 'slug', 'all_text', 'with_milk', 'name', 'review_date'
        self.df = self.df.drop(columns = ['slug', 'all_text', 'with_milk', 'name', 'review_date'])
    
    def drop_nas(self):
        # drop rows with NaNs in cleaned dataframe
        self.df = self.df.dropna(axis = 0)
