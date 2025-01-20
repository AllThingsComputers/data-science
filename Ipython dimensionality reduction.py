# Required Imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Display all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load the dataset
df = pd.read_csv(r"C:\Users\User.user\source\repos\57. AI ML p2\SE4624.csv")
print("Number of Columns:", len(df.columns))
print("Number of Rows:", len(df))
print(df.head())

# Filter for 'User' AccountType
df = df[df['AccountType'] == 'User']

# Select relevant columns
columns = [
    'TenantId', 'TimeGenerated [Local Time]', 'SourceSystem', 'Account', 
    'AccountType', 'Computer', 'EventSourceName', 'Channel', 'Task', 'Level',
    'WorkstationName', 'SourceComputerld', 'EventOriginld', 'MG', 
    'TimeCollected [Local Time]', 'ManagementGroupName', 'Type', '_Resourceld'
]
df = df[columns]

# Save the filtered dataset
output_path = r"C:\Users\User.user\source\repos\57. AI ML p2\new4624sec.csv"
df.to_csv(output_path, index=False)

# Reload the filtered dataset
df_user = pd.read_csv(output_path)
print("Number of Columns:", len(df_user.columns))
print("Number of Rows:", len(df_user))
print(df_user.head())

# Fill missing values with mean for specified columns
def fill_na(numerical_column):
    if numerical_column in df_user.columns:
        df_user[numerical_column].fillna(df_user[numerical_column].mean(), inplace=True)

columns_to_fill = [
    'mths_since_recent_revol_deling', 'num_accts_ever_120_pd', 'num_actv_bc_tl',
    'num_actv_rev_tl', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util'
]
for col in columns_to_fill:
    fill_na(col)

print(df_user.head())

# Filter rows with specific values in the 'Authentication PackageName' column
df_user = df_user[df_user['Authentication PackageName'].isin(['NTLM', 'Kerberos', 'Negotiate'])]

# Add a label column
df_user['Authentication_PackageName_label'] = np.where(df_user['Authentication PackageName'] == 'NTLM', 0, 1)
print(df_user.head())

# Select further relevant columns
columns2 = [
    'TenantId', 'TimeGenerated [Local Time]', 'WorkstationName',
    'SourceComputerld', 'EventOriginld', 'MG', 'TimeCollected [Local Time]',
    'ManagementGroupName', 'Type', '_Resourceld'
]
df_user = df_user[columns2]
print(df_user.head())

# Random Forest Feature Importance
X = df_user[['Task', 'Level', 'EventID', 'LogonType']]
y = df_user['Authentication_PackageName_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.33)

model = RandomForestClassifier()
model.fit(X_train, y_train)

features = ['Task', 'Level', 'EventID', 'LogonType']
feature_df = pd.DataFrame({
    "Importance": model.feature_importances_,
    "Features": features
})

sns.set()
plt.bar(feature_df["Features"], feature_df["Importance"])
plt.xticks(rotation=90)
plt.title("Random Forest Model Feature Importance")
plt.show()

# PCA for Dimensionality Reduction
features2 = ['Task', 'Level', 'LogonType']
X = df_user[features2]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
pca.fit(X_scaled)
X_components = pca.transform(X_scaled)

components_df = pd.DataFrame({
    'component_one': X_components[:, 0],
    'component_two': X_components[:, 1],
    'component_three': X_components[:, 2]
})
print(components_df.head())

# Scatter plot of PCA components
labels = df_user['LogonType']
color_dict = {0: 'Red', 1: 'Blue'}

fig, ax = plt.subplots(figsize=(7, 5))
sns.set()
default_color = 'black'

for i in np.unique(labels):
    index = np.where(labels == i)
    if i in color_dict.keys():
        ax.scatter(
            components_df['component_one'].iloc[index],
            components_df['component_two'].iloc[index],
            c=color_dict[i], s=10, label=i
        )
    else:
        ax.scatter(
            components_df['component_one'].iloc[index],
            components_df['component_two'].iloc[index],
            c=default_color, s=10, label=i
        )

plt.xlabel("1st Component", fontsize=14)
plt.ylabel("2nd Component", fontsize=14)
plt.title('Scatter plot of Principal Components')
plt.legend()
plt.show()
