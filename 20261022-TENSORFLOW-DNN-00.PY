import tensorflow as tf
import numpy as np

# 3. 새로운 입력 데이터 및 실제 정답(Label) 준비
new_data = np.array([[1.0, 0.5], [0.2, 0.8]])
true_labels = np.array([[1.0], [0.0]]) # 평가를 위한 실제 정답 데이터 (2개 샘플)

# 1. 간단한 신경망 모델 정의 (입력 2개 -> 은닉층 4개 -> 출력 1개)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)), # 은닉층
    tf.keras.layers.Dense(1, activation='sigmoid')                # 출력층
])

# 2. 모델 컴파일 (추론 전 설정 - 정확도 측정을 위해 metrics 추가)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. 추론 알고리즘 실행 (predict)
predictions = model.predict(new_data)

model.fit(new_data, true_labels, epochs=5)

# 6. 최종 평가 (실제 정답 데이터인 true_labels를 전달)
loss, accuracy = model.evaluate(new_data, true_labels, verbose=0)
