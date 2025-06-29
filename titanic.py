import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
df = pd.read_csv("C:\\Users\\asus\\OneDrive\\Desktop\\K23GP\\summer training\\titanic.csv")

# 1. Check basic structure of dataset
print("Dataset shape:", df.shape)
print("Column names:", df.columns.tolist())
print("Data types:\n", df.dtypes)
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Insight:
# Dataset has 887 rows and 8 columns. No missing or duplicate data. All data types are as expected.

# 2. Count unique values in each column
print("\nUnique values in each column:\n", df.nunique())

# Insight:
# Useful to understand categorical diversity. Sex and Pclass have limited unique values.

# 3. Summary statistics for numerical columns
print("\nStatistical Summary:\n", df.describe())

# Insight:
# Age has a mean around 29, and Fare has a large range and standard deviation, indicating outliers.

# 4. Central tendency and spread
print("\nMean:\n", df.mean(numeric_only=True))
print("\nMedian:\n", df.median(numeric_only=True))
print("\nMode:\n", df.mode(numeric_only=True).iloc[0])
print("\nStandard Deviation:\n", df.std(numeric_only=True))
print("\nVariance:\n", df.var(numeric_only=True))
print("\nMinimum:\n", df.min(numeric_only=True))
print("\nMaximum:\n", df.max(numeric_only=True))

# Insight:
# Central tendency values are close in Age, indicating less skew. Fare has outliers.

# 5. Survival distribution
sns.countplot(x='Survived', data=df, palette="magma")
plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Insight:
# Around 60% did not survive, while 40% did.

# 6. Survival based on gender
sns.countplot(x='Sex', hue='Survived', data=df, palette="cubehelix")
plt.title("Survival by Sex")
plt.show()

# Insight:
# Female passengers had a higher chance of survival than males.

# 7. Survival based on passenger class
sns.countplot(x='Pclass', hue='Survived', data=df, palette="coolwarm")
plt.title("Survival by Passenger Class")
plt.show()

# Insight:
# Passengers in 1st class survived more frequently than those in 3rd.

# 8. Age distribution
sns.histplot(df['Age'], bins=30, kde=True, color='coral')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()

# Insight:
# Most passengers were aged between 20–40. Young adults were the majority.

# 9. Age vs survival
sns.boxplot(x='Survived', y='Age', data=df, palette="autumn")
plt.title("Age Distribution by Survival")
plt.show()

# Insight:
# Survivors had a slightly lower median age than non-survivors.

# 10. Fare vs survival
sns.boxplot(x='Survived', y='Fare', data=df, palette="spring")
plt.title("Fare Distribution by Survival")
plt.show()

# Insight:
# Higher fare paid correlates with higher survival.

# 11. Pairplot with main numerical features
sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']], hue='Survived', palette="husl")
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.show()

# Insight:
# Clear patterns in age, class, and fare relative to survival.

# 12. Pairplot including Sex (encoded)
df_encoded = df.copy()
df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})

sns.pairplot(df_encoded[['Survived', 'Sex', 'Age', 'Fare']], hue='Survived', palette="Set2")
plt.suptitle("Pairplot Including Gender", y=1.02)
plt.show()

# Insight:
# Women (Sex=1) had higher survival, showing strong gender-based separation.

# 13. Scatterplot Age vs Fare
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, palette="plasma")
plt.title("Age vs Fare by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

# Insight:
# Survivors generally paid higher fares.

# 14. Scatterplot Age vs Siblings/Spouses
sns.scatterplot(x='Age', y='Siblings/Spouses Aboard', hue='Survived', data=df, palette="cividis")
plt.title("Age vs Siblings/Spouses Aboard by Survival")
plt.xlabel("Age")
plt.ylabel("Siblings/Spouses Aboard")
plt.show()

# Insight:
# People with 1–2 siblings or spouses had better survival chances.

# 15. Scatterplot Age vs Parents/Children
sns.scatterplot(x='Age', y='Parents/Children Aboard', hue='Survived', data=df, palette="rocket")
plt.title("Age vs Parents/Children Aboard by Survival")
plt.xlabel("Age")
plt.ylabel("Parents/Children Aboard")
plt.show()

# Insight:
# Young children and those with parents had higher survival.

# 16. Correlation heatmap
numerical = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numerical.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='Spectral', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# Insight:
# Survival positively correlated with Sex (female), Fare; negatively with Pclass.

# 17. Function to detect outliers using IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    print(f"\n{column} Outliers:")
    print(f"Lower Bound: {lower:.2f}, Upper Bound: {upper:.2f}")
    print(f"Number of Outliers in {column}: {outliers.shape[0]}")
    return outliers

# Check outliers in Fare and Age
outliers_fare = detect_outliers_iqr(df, 'Fare')
outliers_age = detect_outliers_iqr(df, 'Age')

# Insight:
# Fare has several high-value outliers; Age has few.

# 18. Boxplot to show both Age and Fare outliers
sns.boxplot(data=df[['Age', 'Fare']], palette='YlGnBu')
plt.title("Boxplot to Detect Outliers")
plt.show()

# Insight:
# Fare has extreme high values; Age distribution is tighter.

# Final Insight:
# Key features for survival: Gender (female), Pclass (1st), Fare (high), Age (young), family members.

# 19. Classification using KNN and Decision Tree
# Encode categorical column
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Prepare features and target
X = df.drop(['Survived', 'Name', 'Ticket'], axis=1, errors='ignore')
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_acc = accuracy_score(y_test, y_pred_knn)
print(f"\nKNN Accuracy: {knn_acc*100:.2f}%")

# Insight:
# KNN works by checking closest neighbors in feature space. Accuracy depends on scaling and distance.

# Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {dt_acc*100:.2f}%")

# Insight:
# Decision Trees split data based on rules. They handle categorical and non-linear data well.

if knn_acc > dt_acc:
    print("KNN performed better on this dataset.")
elif dt_acc > knn_acc:
    print("Decision Tree performed better on this dataset.")
else:
    print("Both models performed equally well.")

# Final Insight on Accuracy:
# KNN Accuracy: shows performance based on proximity.
# Decision Tree Accuracy: typically better with categorical and rule-based patterns.
# Choose model based on explainability or accuracy needs.
