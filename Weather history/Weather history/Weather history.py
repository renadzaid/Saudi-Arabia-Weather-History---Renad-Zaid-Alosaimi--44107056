import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Data/Original Data/weather-sa-2017-2019-clean.csv")
df.drop(columns=["Unnamed: 0", "date", "time"], inplace=True)
df.drop_duplicates(inplace=True)

df["humidity"] = df["humidity"].str.rstrip('%').astype(float)
df.dropna(subset=["humidity", "barometer"], inplace=True)

top_classes = df["weather"].value_counts().nlargest(3).index.tolist()
df = df[df["weather"].isin(top_classes)]

label_enc_weather = LabelEncoder()
df["weather"] = label_enc_weather.fit_transform(df["weather"])
weather_classes = label_enc_weather.classes_

label_enc_city = LabelEncoder()
df["city"] = label_enc_city.fit_transform(df["city"])

df = df.groupby("city").apply(lambda x: x.sample(n=min(500, len(x)), random_state=42)).reset_index(drop=True)

features = ['city', 'month', 'hour', 'temp', 'wind', 'humidity', 'barometer', 'visibility']
X = df[features]
y = df["weather"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pd.DataFrame(X_train_scaled, columns=features).to_csv("data/preprocessed/X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=features).to_csv("data/preprocessed/X_test.csv", index=False)
y_train.to_csv("data/preprocessed/Y_train.csv", index=False)
y_test.to_csv("data/preprocessed/Y_test.csv", index=False)

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, class_weight='balanced'),
    "NaiveBayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    "ANN": MLPClassifier(max_iter=2000, random_state=42, early_stopping=True)
}


for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    pd.DataFrame({"Prediction": y_pred}).to_csv(f"data/Results/prediction_{name}.csv", index=False)

    print(classification_report(y_test, y_pred, target_names=weather_classes, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=weather_classes, yticklabels=weather_classes)
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"data/Results/confusion_matrix_{name}.png")
    plt.close()
