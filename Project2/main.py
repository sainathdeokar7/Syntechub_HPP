
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# 2) Load the Iris dataset

from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data        # feature matrix (150 rows × 4 features)
y = iris.target      # labels (0=setosa, 1=versicolor, 2=virginica)

# Optional: convert to DataFrame for plotting
df_iris = pd.DataFrame(X, columns=iris.feature_names)
df_iris['species'] = pd.Categorical.from_codes(y, iris.target_names)


# 3) Exploratory Visualization
sns.pairplot(df_iris, hue='species')
plt.suptitle("Iris Data Pairplot", y=1.02)
plt.show()

# 4) Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 5) Train SVM Classifier
model = SVC(kernel='linear')  # Linear SVM classifier
model.fit(X_train, y_train)

# 6) Evaluate Model
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")

print("\nDetailed classification report:\n")
print(classification_report(y_test, y_pred,
      target_names=iris.target_names))

# 7) Make New Predictions
X_new = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.7, 4.0, 1.5],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model.predict(X_new)
print("\nPredictions for new samples:", predictions)
for idx, p in enumerate(predictions):
    print(f" Sample {idx} --> {iris.target_names[p]}")

# 8) Save & Reload Model
import pickle
with open('iris_svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# to reload:
# with open('iris_svm_model.pkl','rb') as f:
# loaded_model = pickle.load(f)
#     print(loaded_model.predict(X_new))

# End of script