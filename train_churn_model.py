import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# 1. Load your dataset
df = pd.read_csv('Churn_Modelling.csv')

# 2. Select features and label (adjust column names as per your dataset)
features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
            'NumOfProducts', 'HasCrCard', 'IsActiveMember']  # update if needed
label = 'Exited'  # 1 = churned, 0 = not churned

X = df[features]
y = df[label]

# 3. Encode categorical features
# Gender (LabelEncoder)
label_encoder_gender = LabelEncoder()
X['Gender'] = label_encoder_gender.fit_transform(X['Gender'])

# Geography (OneHotEncoder)
# Geography (OneHotEncoder)
onehot_encoder_geo = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
geo_encoded = onehot_encoder_geo.fit_transform(X[['Geography']])
geo_cols = onehot_encoder_geo.get_feature_names_out(['Geography'])
geo_df = pd.DataFrame(geo_encoded, columns=geo_cols, index=X.index)
X = X.drop('Geography', axis=1)
X = pd.concat([X, geo_df], axis=1)

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Build your Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# 7. Save artifacts for inference
model.save('model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder_gender.pkl', 'wb') as f:
    pickle.dump(label_encoder_gender, f)
with open('onehot_encoder_geo.pkl', 'wb') as f:
    pickle.dump(onehot_encoder_geo, f)

print('Training complete! Model and encoders saved.')
