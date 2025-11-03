import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. DATA GENERATION & PREPROCESSING ---
from google.colab import drive
drive.mount("/content/drive")
df = pd.read_csv("/content/drive/MyDrive/spam.csv", encoding = "ISO-8859-1")
df.rename(columns = {"v1": "Label", "v2": "Message"}, inplace = True)

print("--- Initial Data Snapshot ---")
print(df.head())
print("-" * 35)

# 1.1 Text Cleaning (As provided by the user)
df_col_message = df["Message"].str.lower()
# Using re.sub for robust regex replacement (similar to the str.replace with regex=True)
df_col_message = df_col_message.apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
df["Message"] = df_col_message

# 1.2 Label Encoding
df["Label"] = df["Label"].replace({"ham" : 1, "spam" : -1})
print("\n--- Cleaned & Encoded Data Snapshot ---")
print(df.head())
print("-" * 35)

# 1.3 Split the Dataset
X = df["Message"]  # features (text)
y = df["Label"]    # targets (1 or -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")
print("-" * 35)


# --- 2. FEATURE EXTRACTION ---

# 2.1 Initialize TfidfVectorizer
# TfidfVectorizer is chosen as it weighs word importance.
vectorizer = TfidfVectorizer()

# 2.2 Fit and Transform the Training Data
# The vectorizer learns the vocabulary and IDF weights ONLY from the training data.
X_train_vec = vectorizer.fit_transform(X_train)

# 2.3 Transform the Test Data
# The learned vocabulary and IDF weights are APPLIED to the test data.
X_test_vec = vectorizer.transform(X_test)

print("--- Feature Extraction Complete ---")
print(f"Vocabulary Size (Features): {len(vectorizer.get_feature_names_out())}")
print(f"Training Feature Matrix Shape: {X_train_vec.shape}")
print(f"Test Feature Matrix Shape: {X_test_vec.shape}")
print("-" * 35)


# --- 3. MODEL TRAINING ---

# 3.1 Initialize the Model
model = MultinomialNB()

# 3.2 Train the Model
# The model learns the probabilities of each feature (word) belonging to the 'spam' vs 'ham' class.
model.fit(X_train_vec, y_train)

print("--- Model Training Complete (Multinomial Naive Bayes) ---")
print("-" * 35)


# --- 4. EVALUATION ---

# 4.1 Prediction
y_pred = model.predict(X_test_vec)

# 4.2 Compute Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Spam (-1)', 'Ham (1)'])

print("--- Model Evaluation Results ---")
print(f"Accuracy: {accuracy:.4f}\n")
print("Confusion Matrix:")
# Rows are True Labels, Columns are Predicted Labels
# [[True Negative, False Positive]
#  [False Negative, True Positive]]
print(conf_matrix)

print("\nClassification Report (Detailing Precision, Recall, F1-Score):")
print(class_report)

print("-" * 35)
print("Project Execution Summary: All steps (Data Prep, Vectorization, Training, Evaluation) are complete.")