import re, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

from FreqEst import FrequencyEstimator

class CookieDataLoader:
    def __init__(self, csv_path='classified_cookies.csv', test_size=0.1, random_state=42):
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state
        self._load_and_split()

    def _load_and_split(self):
        rows = []
        with open(self.csv_path, 'r', encoding='utf-8', errors='replace') as file:
            for line in file:
                parts = line.strip().split(',', 6)
                row = parts[1:] if len(parts) > 1 else []
                while len(row) < 6:
                    row.append('')
                rows.append(row[:6])

        dataframe = pd.DataFrame(
            rows,
            columns=['Cookie_Name', 'Category', 'Provider', 'Site_Found', 'Duration', 'Description'],
        )
        dataframe['Category'] = dataframe['Category'].replace({'Functional': 'Preferences'})
        valid_categories = ['Marketing', 'Statistics', 'Necessary', 'Preferences']
        dataframe = dataframe[dataframe['Category'].isin(valid_categories)]
        self.df = dataframe

        train, test = train_test_split(
            dataframe,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=dataframe['Category'],
        )

        columns = ['Cookie_Name', 'Provider', 'Site_Found', 'Duration']
        self.X_train_raw, self.y_train = train[columns], train['Category']
        self.X_test_raw, self.y_test = test[columns], test['Category']

    @staticmethod
    def prepare_features(dataframe):
        def clean_domain(value):
            if pd.isna(value):
                return ''
            return str(value).replace('.', ' ').replace('-', ' ')

        def parse_duration(value):
            if pd.isna(value) or str(value).strip() == '':
                return np.nan
            text = str(value).lower().strip()
            if 'session' in text or 'never' in text:
                return np.nan

            units = {
                'year': 365 * 24 * 60,
                'month': 30 * 24 * 60,
                'day': 24 * 60,
                'hour': 60,
                'minute': 1,
                'second': 1 / 60,
            }
            total_minutes = 0.0
            found_unit = False
            for unit, multiplier in units.items():
                match = re.search(r'(\d+(?:\.\d+)?)\s*' + unit, text)
                if match:
                    total_minutes += float(match.group(1)) * multiplier
                    found_unit = True
            return total_minutes if found_unit else np.nan

        return pd.DataFrame({
            'Cookie_Name': dataframe['Cookie_Name'].fillna('').astype(str),
            'Provider': dataframe['Provider'].apply(clean_domain),
            'Site_Found': dataframe['Site_Found'].apply(clean_domain),
            'Duration': dataframe['Duration'].apply(parse_duration),
        }, index=dataframe.index)

    @staticmethod
    def encode_utf8(input, max_length=64):
        data = input.encode("utf-8")[:max_length]

        x = np.zeros(max_length, dtype=np.float32)
        x[:len(data)] = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 255.0

        return x

loader = CookieDataLoader()

X_train = loader.prepare_features(loader.X_train_raw)
X_test = loader.prepare_features(loader.X_test_raw)
y_train = loader.y_train
y_test = loader.y_test

X_train_name_necessary = np.stack([
    loader.encode_utf8(text)
    for text in X_train.loc[y_train == "Necessary", "Cookie_Name"]
])

X_train_name_preferences = np.stack([
    loader.encode_utf8(text)
    for text in X_train.loc[y_train == "Preferences", "Cookie_Name"]
])

X_train_name_statistics= np.stack([
    loader.encode_utf8(text)
    for text in X_train.loc[y_train == "Statistics", "Cookie_Name"]
])

X_train_name_marketing = np.stack([
    loader.encode_utf8(text)
    for text in X_train.loc[y_train == "Marketing", "Cookie_Name"]
])

X_train_site_provider_necessary = np.stack([
    loader.encode_utf8(text)
    for text in X_train.loc[y_train == "Necessary", "Provider"]
])

X_train_site_provider_preferences = np.stack([
    loader.encode_utf8(text)
    for text in X_train.loc[y_train == "Preferences", "Provider"]
])

X_train_site_provider_statistics = np.stack([
    loader.encode_utf8(text)
    for text in X_train.loc[y_train == "Statistics", "Provider"]
])

X_train_site_provider_marketing = np.stack([
    loader.encode_utf8(text)
    for text in X_train.loc[y_train == "Marketing", "Provider"]
])

X_train_site_site_found_necessary = np.stack([
    loader.encode_utf8(text)
    for text in X_train.loc[y_train == "Necessary", "Site_Found"]
])

X_train_site_site_found_preferences = np.stack([
    loader.encode_utf8(text)
    for text in X_train.loc[y_train == "Preferences", "Site_Found"]
])

X_train_site_site_found_statistics = np.stack([
    loader.encode_utf8(text)
    for text in X_train.loc[y_train == "Statistics", "Site_Found"]
])

X_train_site_site_found_marketing = np.stack([
    loader.encode_utf8(text)
    for text in X_train.loc[y_train == "Marketing", "Site_Found"]
])

X_train_site_duration_necessary = X_train.loc[
    y_train == "Necessary", "Duration"
].fillna(0).to_numpy(dtype=np.float32)

X_train_site_duration_preferences = X_train.loc[
    y_train == "Preferences", "Duration"
].fillna(0).to_numpy(dtype=np.float32)

X_train_site_duration_statistics = X_train.loc[
    y_train == "Statistics", "Duration"
].fillna(0).to_numpy(dtype=np.float32)

X_train_site_duration_marketing = X_train.loc[
    y_train == "Marketing", "Duration"
].fillna(0).to_numpy(dtype=np.float32)


print("Finnished loading data")
FrequencyEstimator(X_train_name_marketing).transform_to_dct()
