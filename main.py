import os
import shutil
import time
from datetime import date, datetime
import hashlib
import json
import re
import tabula
import pandas as pd
import traceback
from currency_converter import CurrencyConverter, ECB_URL
import logging
from collections import defaultdict

class UtilityFunctions:
	def __init__(self, currency_converter):
		self.currency_converter = currency_converter

	def _convert_currency(self, amount, base_currency, target_currency, specified_date):
		try:
			extract_date = datetime.strptime(specified_date, '%d-%m-%Y').date()
			converted_amount = self.currency_converter.convert(amount, base_currency, target_currency, extract_date)
			return round(converted_amount, 2)
		except Exception as e:
			logging.error(f"Error converting currency: {str(e)}")
		return None # Return None if an exception occured

	def get_currency_rate(self, row):
		base_currency = row['Currency']
		target_currency = "DKK"
		return self._convert_currency(1.00, base_currency, target_currency, row['Date'])

	def calculate_target_currency_rate(self, row):
		specified_date = row['Date']
		base_currency = row['Currency']
		target_currency = "DKK"
		amount = row['Amount_currency']
		return self._convert_currency(amount, base_currency, target_currency, specified_date)

class DataImport:
	def __init__(self, file_path, bank, currency_converter=None):
		self.file_path = file_path
		self.bank = bank
		self.data = None
		self.currency_converter = currency_converter or CurrencyConverter(ECB_URL, 
															fallback_on_wrong_date=True, 
															fallback_on_missing_rate=True)
		self.load_methods = {
			'Danske Bank': self.load_DanskeBank,
			'Wise': self.load_Wise,
			'Norwegian': self.load_Norwegian,
			'Forbrugsforeningen': self.load_Forbrugsforeningen,
			'Lunar': self.load_Lunar
		}

	def load_data(self):
		if self.bank not in self.load_methods:
			raise ValueError(f"No data load method for bank: {self.bank}")

		load_method = self.load_methods[self.bank]
		load_method()

	def _common_preprocessing(self, df):
		df['UniqueID'] = df.apply(lambda row: 
			hashlib.md5(f"{row['Date']}_{row['Description']}_{row['Amount']}_{row['UniqueCol']}".encode()).hexdigest(), 
			axis=1
			)

	def load_Lunar(self):
		dtype_map = {
			'Dato': 'Date',
			'Tekst': 'Description',
			'Beløb': 'Amount',
			'Tid': 'UniqueCol'
		}

		df = pd.read_csv(
			self.file_path,
			delimiter=',',
			encoding='utf-8',
			usecols=list(dtype_map.keys())
		)

		df = df.rename(columns=dtype_map)

		df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d').dt.strftime('%d-%m-%Y')
		df['Amount'] = (
			df['Amount'].str.replace('.', '', regex=False)
			.str.replace(',', '.', regex=False)
			.astype(float)
		)

		df['Bank'] = 'Lunar'
		df['Amount_currency'] = df['Amount']
		df['Currency'] = 'DKK'
		df['Currency_Rate'] = 1

		self._common_preprocessing(df)

		selected_columns = [
			'Date', 'Description', 'Amount', 'Amount_currency', 
			'Currency', 'Currency_Rate', 'Bank', 'UniqueID'
		]
		self.data = df[selected_columns].to_dict(orient='records')

	def load_DanskeBank(self):
		dtype_map = {
			'Dato': 'Date',
			'Tekst': 'Description',
			'Beløb': 'Amount',
			'Saldo': 'UniqueCol',
			'Status': 'Status'
		}

		df = pd.read_csv(
			self.file_path,
			delimiter=';',
			encoding='ISO-8859-1',
			usecols=list(dtype_map.keys())
		)

		df = df[df['Status'] != 'Slettet']
		df = df[df['Status'] != 'Venter']

		df = df.rename(columns=dtype_map)

		df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y').dt.strftime('%d-%m-%Y')
		df['Amount'] = (
        	df['Amount'].str.replace('.', '', regex=False)
        	.str.replace(',', '.', regex=False)
        	.astype(float)
    	)
		
		df['Bank'] = 'Danske Bank'
		df['Amount_currency'] = df['Amount']
		df['Currency'] = 'DKK'
		df['Currency_Rate'] = 1

		self._common_preprocessing(df)

		selected_columns = [
			'Date', 'Description', 'Amount', 'Amount_currency', 
			'Currency', 'Currency_Rate', 'Bank', 'UniqueID'
		]
		self.data = df[selected_columns].to_dict(orient='records')


	def load_Wise(self):
		df = pd.read_csv(self.file_path, 
						 delimiter=',', 
						 encoding='utf-8')
		valid_rows = (df['Status'] != 'CANCELLED') & \
					 (df['Target amount (after fees)'] != 0) & \
					 (df['Source amount (after fees)'] != 0)
		df = df[valid_rows]

		df['Date'] = pd.to_datetime(df['Finished on'], 
									format='%Y-%m-%d %H:%M:%S').dt.strftime('%d-%m-%Y')
		
		df['Description'] = df.apply(
			lambda row: row['Source name'] if row['Direction'] == 'IN' else (
				row['Target name'] if row['Direction'] == 'OUT' else 'Internal transfer'),
				axis=1)
		df['Amount_currency'] = df.apply(
			lambda row: row['Source amount (after fees)'] if row['Direction'] == 'IN' else (
			row['Target amount (after fees)'] * -1 if row['Direction'] == 'OUT' else 0), 
			axis=1)
		df['Currency'] = df.apply(
			lambda row: row['Source currency'] if row['Direction'] == 'IN' else (
			row['Target currency'] if row['Direction'] == 'OUT' else row['Target currency']), 
			axis=1)

    	# Create bank identifier
		df['Bank'] = 'Wise'

		#Create UniqueCol
		df['UniqueCol'] = df['ID']
 
		#create DKK_rate
		utility_functions_instance = UtilityFunctions(self.currency_converter)
		df['Amount'] = df.apply(utility_functions_instance.calculate_target_currency_rate, axis=1)
		df['Currency_Rate'] = df.apply(utility_functions_instance.get_currency_rate, axis=1)

		self._common_preprocessing(df)
		
		selected_columns = ['Date', 'Description', 'Amount', 'Amount_currency', 
							'Currency', 'Currency_Rate', 'Bank', 'UniqueID']
		self.data = df[selected_columns].to_dict(orient='records')

	def load_Norwegian(self):
		columns_to_read = {
			'TransactionDate': 'Date',
			'Text': 'Description',
			'Amount': 'Amount',
			'Type': 'Type',
			'Currency Amount': 'Amount_currency',
			'Currency': 'Currency',
			'Currency Rate': 'Currency_Rate'
		}
		df = pd.read_excel(self.file_path, usecols=list(columns_to_read.keys()))
		df = df.rename(columns=columns_to_read)

		df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%d-%m-%Y')
		if df['Date'].isnull().any():
			logging.warning(f"Some dates in {self.file_path} could not be parsed.")

		#Create UniqueCol
		df['UniqueCol'] = df['Type']
		self._common_preprocessing(df)
		df['Bank'] = 'Norwegian'
		
		df = df[df['Type'] != "Reserveret"]
		selected_columns = ['Date', 'Description', 'Amount', 'Amount_currency', 
							'Currency', 'Currency_Rate', 'UniqueID', 'Bank']	
		self.data = df[selected_columns].to_dict(orient='records')

	def load_Forbrugsforeningen(self):
		pdf_file = self.file_path
		pdf_dfs = tabula.read_pdf(pdf_file, pages='all', multiple_tables=True, encoding='utf-8')
		concatenated_dfs = []

		for single_df in pdf_dfs:
			if {'Dato', 'Posteringstekst', 'Beløb', 'Valuta', 'Saldo'}.issubset(single_df.columns):
				renamed_df = single_df.rename(columns={
					'Dato': 'Date',
					'Posteringstekst': 'Description',
					'Beløb': 'Amount',
					'Valuta': 'Currency',
					'Saldo': 'UniqueCol'
				})
				concatenated_dfs.append(renamed_df)

		if not concatenated_dfs:
			logging.warning(f"No data was extracted from {pdf_file}")
			return pd.DataFrame()

		df = pd.concat(concatenated_dfs, ignore_index=True)
		df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.strftime('%d-%m-%Y')
		df['Amount'] = df['Amount'].str.replace('.', '').str.replace(',', '.', regex=False).astype(float)
		self._common_preprocessing(df)
		df['Bank'] = 'Forbrugsforeningen'
		df['Amount_currency'] = df['Amount']
		df['Currency_Rate'] = 1
        
		selected_columns = ['Date', 'Description', 'Amount', 'Amount_currency', 
							'Currency', 'Currency_Rate', 'Bank', 'UniqueID']

		self.data = df[selected_columns].to_dict(orient='records')


class Categorization:
	def __init__(self, categorization_rules_path, output_csv_path):
		self.categorization_rules_path = categorization_rules_path
		self.categorization_rules = self.load_categorization_rules()
		self.output_csv_path = output_csv_path

	def load_categorization_rules(self):
		try:
			with open(self.categorization_rules_path, 'r') as json_file:
				categorization_rules = json.load(json_file)
			return categorization_rules
		except FileNotFoundError:
			logging.error(f"File not found: {self.categorization_rules_path}")
			return []
		except Exception as e:
			logging.error(f"An error occurred: {str(e)}")
			return []

	def append_categorization_rules(self, new_rules):
		self.categorization_rules.extend(new_rules)

		try:
			with open(self.categorization_rules_path, 'w') as json_file:
				json.dump(self.categorization_rules, json_file, indent=4)
			print(f"Categorization rules appended to {self.categorization_rules_path}")
		except Exception as e:
			logging.error(f"An error occurred: {str(e)}")

	def categorize_transactions(self, transactions):
		categorized_transactions = []

		for transaction in transactions:
			categorized_transaction = transaction.copy()

			for rule in self.categorization_rules:
				keyword_orig = rule['Keyword']
				keyword_without_numbers = re.sub(r'^\d+', '', keyword_orig)
				keyword = keyword_without_numbers.strip()
				main_category = rule['Main Category']
				sub_category = rule['Sub Category']

				if transaction['Description'].lower().startswith(keyword.lower()):
					categorized_transaction['Main Category'] = main_category
					categorized_transaction['Sub Category'] = sub_category
					break

			if 'Main Category' not in categorized_transaction:
				categorized_transaction['Main Category'] = 'Uncategorized'
			if 'Sub Category' not in categorized_transaction:
				categorized_transaction['Sub Category'] = 'Uncategorized'

			categorized_transactions.append(categorized_transaction)

		return categorized_transactions

	def update_existing_category(self, output_csv_path, categorization_rules_path):
		#load existing data from the output CSV file
		with open(categorization_rules_path, 'r') as json_file:
			categorization_rules = json.load(json_file)
		existing_data = pd.read_csv(output_csv_path, encoding='utf-8', sep=';', decimal=',')
		categorization_dict = {rule['Keyword'].lower(): rule for rule in self.categorization_rules}

		def categorize_description(desc):
			for keyword, rule in categorization_dict.items():
				desc_without_numbers = re.sub(r'^\d+', '', desc)
				stripped_desc = desc_without_numbers.strip()
				if stripped_desc.lower().startswith(keyword.lower()):
					return rule['Main Category'], rule['Sub Category']
			return 'Uncategorized', 'Uncategorized'

		existing_data[['Main Category', 'Sub Category']] = existing_data['Description'].apply(categorize_description).apply(pd.Series)

		# Apply the categorize_description function to the 'Description' column
		existing_data[['Main Category', 'Sub Category']] = existing_data['Description'].apply(categorize_description).apply(pd.Series)

		# Save the updated data back to the output CSV file
		existing_data.to_csv(output_csv_path, index=False, sep=";", decimal=",", encoding='utf-8')

		print(f"Categories updated in {output_csv_path}")

class Testing:

	@staticmethod
	def load_json_data(json_file_path):
		try:
			with open(json_file_path, 'r') as file:
				return json.load(file)
		except Exception as e:
			print(f"An error occurred while loading JSON data from {json_file_path}: {str(e)}")
			return []

	@staticmethod
	def check_categories_and_duplicates(json_file_path):
		print(f"Checking file for conflicting categories: {json_file_path}")
		data = Testing.load_json_data(json_file_path)

		keywords_to_main_categories = {}
		keywords_to_sub_categories = {}
		conflicting_categories = defaultdict(list)

		for item in data:
			keyword = item.get('Keyword')
			main_category = item.get('Main Category')
			sub_category = item.get('Sub Category')

			if keyword is None:
				continue

			# Handle main category conflicts
			if keyword in keywords_to_main_categories and keywords_to_main_categories[keyword] != main_category:
				conflicting_categories[keyword] += [keywords_to_main_categories[keyword], main_category]
			else:
				keywords_to_main_categories[keyword] = main_category

			# Handle sub-category conflicts
			if keyword in keywords_to_sub_categories and keywords_to_sub_categories[keyword] != sub_category:
				conflicting_categories[keyword] += [keywords_to_sub_categories[keyword], sub_category]
			else:
				keywords_to_sub_categories[keyword] = sub_category

		return dict(conflicting_categories)

	@staticmethod
	def find_redundant_keywords(json_file_path):
		print(f"Checking file for redundant keywords: {json_file_path}")
		data = Testing.load_json_data(json_file_path)

		keywords_count = defaultdict(list)
		redundant_keywords = defaultdict(list)

		for item in data:
			keyword = item.get('Keyword')
			if keyword is None:
				continue

			# Increase count or mark as redundant
			if keyword in keywords_count:
				redundant_keywords[keyword].append(item)
			keywords_count[keyword].append(item)


		# Only return keywords that are redundant (appear more than once)
		redundant_keywords = {k: v for k, v in redundant_keywords.items() if len(v) > 1}

		return redundant_keywords

class DataTransformer:
	def __init__(self, output_file_path):
		self.output_file_path = output_file_path

	def process_and_export_data(self, transactions):
		try:
			if os.path.isfile(self.output_file_path):
				# Load the existing CSV file
				existing_data = pd.read_csv(self.output_file_path, encoding='utf-8', sep=';', decimal=',')

				# Create a DataFrame from the new transactions
				new_data = pd.DataFrame(transactions)

				# Convert 'UniqueID' columns to strings in both DataFrames
				new_data['UniqueID'] = new_data['UniqueID'].astype(str)
				existing_data['UniqueID'] = existing_data['UniqueID'].astype(str)

				# Check for duplicates based on the 'UniqueID' column
				duplicates = new_data[new_data['UniqueID'].isin(existing_data['UniqueID'])]

				if not duplicates.empty:
					# Remove duplicates from the new_data DataFrame
					new_data = new_data[~new_data['UniqueID'].isin(existing_data['UniqueID'])]
					print(f"Removed {len(duplicates)} duplicate rows from new data.")

				# Concatenate the existing data and new data
				combined_data = pd.concat([existing_data, new_data], ignore_index=True)

				# Write the combined data to the same CSV file
				combined_data.to_csv(self.output_file_path, index=False, sep=';', decimal=',')

				print(f"Data appended to {self.output_file_path}")
			else:
				print(f"Creating a new CSV file: {self.output_file_path}")
				# If the file doesn't exist, create a new one
				df = pd.DataFrame(transactions)
				df.to_csv(self.output_file_path, index=False, encoding='utf-8', sep=';', decimal=',')

				print(f"Data exported to a new file: {self.output_file_path}")

		except Exception as e:
			print(f"An error occurred in process_and_export_data: {str(e)}")

	def backup_csv_file(self):
		try:
			# Check if the output file exists
			if not os.path.isfile(self.output_file_path):
			    print(f"CSV file '{self.output_file_path}' does not exist. Skipping backup.")
			    return

			# Check if the output file is empty
			if os.path.getsize(self.output_file_path) == 0:
			    print(f"CSV file '{self.output_file_path}' is empty. Skipping backup.")
			    return

			# Create a 'backup' subfolder in the directory containing the output file, if it doesn't exist
			backup_folder = os.path.join(os.path.dirname(self.output_file_path), 'backup')
			if not os.path.exists(backup_folder):
				os.makedirs(backup_folder)

			# Extract the filename and extension from the output file path
			file_name = os.path.basename(self.output_file_path)
			name, ext = os.path.splitext(file_name)

			# Create a backup file name by adding a timestamp to the original file name
			backup_file_name = f'{name}_backup_{time.strftime("%Y%m%d%H%M%S")}{ext}'
			backup_file_path = os.path.join(backup_folder, backup_file_name)

			# Copy the original CSV file to the backup location
			shutil.copyfile(self.output_file_path, backup_file_path)

			print(f"Backup created: {backup_file_path}")
		except Exception as e:
			print(f"An error occurred while creating a backup: {str(e)}")


def configure_logging():
    logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')

def main():	
	configure_logging()
	file_path = 'C:/Users/smrie/Downloads/eksport.pdf'
	bank = 'Forbrugsforeningen'
	categorization_rules_path = 'categorization_rules.json'
	output_file_path = 'output.csv'
	data_transformer = DataTransformer(output_file_path=output_file_path)
	data_transformer.backup_csv_file()
	currency_converter_instance = CurrencyConverter(ECB_URL, fallback_on_wrong_date=True, fallback_on_missing_rate=True)
	importer = DataImport(file_path, bank, currency_converter=currency_converter_instance)
	try:
		importer.load_data()
		transactions = importer.data

		categorizer = Categorization(categorization_rules_path, output_file_path)
		#Pre-processing
		categorized_transactions = categorizer.categorize_transactions(transactions)
		transformer = DataTransformer(output_file_path)
		transformer.process_and_export_data(transactions)

		#After manual editing
		categorizer.update_existing_category(output_file_path, categorization_rules_path)

		transformer = DataTransformer(output_file_path)
		transformer.process_and_export_data(transactions)

		tester = Testing()
		conflicting_categories = tester.check_categories_and_duplicates(categorization_rules_path)
		print("Conflicting categories:", conflicting_categories)
		reduntant_keywords = tester.find_redundant_keywords(categorization_rules_path)
		print("Reduntant keywords:", reduntant_keywords)

	except ValueError as e:
		logging.error(e)
	except Exception as e:
		logging.error("An unexpected exception occurred.", exc_info=True)

if __name__ == "__main__":
	main()
