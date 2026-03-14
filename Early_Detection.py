import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
num_employees = 2000


avg_daily_hours = np.random.normal(loc=8.5, scale=1.5, size=num_employees)
meetings_per_day = np.random.poisson(lam=3.5, size=num_employees)
sentiment_score = np.random.uniform(0.1, 0.9, size=num_employees)
days_since_vacation = np.random.randint(10, 300, size=num_employees)

risk_score = (avg_daily_hours * 0.4) + (meetings_per_day * 0.2) - (sentiment_score * 5) + (days_since_vacation * 0.01)
median_risk = np.median(risk_score)

labels = (risk_score > (median_risk + 0.5)).astype(int)

df = pd.DataFrame({
    'avg_daily_hours': avg_daily_hours,
    'meetings_per_day': meetings_per_day,
    'sentiment_score': sentiment_score,
    'days_since_vacation': days_since_vacation,
    'burnout_risk': labels
})

X = df.drop('burnout_risk', axis=1)
y = df['burnout_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Neural Network Model...")
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2), 
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer=Adam(learning_rate=0.005), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, 
                    validation_split=0.2, 
                    epochs=40, 
                    batch_size=32, 
                    verbose=0) 

loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nModel Test Accuracy: {accuracy*100:.2f}%")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='#e74c3c', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='#c0392b', linestyle='--')
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Loss', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='#2ecc71', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#27ae60', linestyle='--')
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Accuracy', fontsize=11)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.suptitle('Deep Learning Training Performance: Burnout Prediction', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()