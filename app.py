import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

# Register custom layer
class UniversalSentenceEncoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False)

    def call(self, inputs):
        return self.encoder(inputs)


model = tf.keras.models.load_model("nlp_model.keras", custom_objects={"UniversalSentenceEncoder": UniversalSentenceEncoder})

def predict_word(word):
    input_tensor = tf.convert_to_tensor([word], dtype=tf.string)
    pred = model.predict(input_tensor)
    class_idx = tf.argmax(pred, axis=1).numpy()[0]
    return {0: "Factual", 1: "Neutral", 2: "Emotional"}.get(class_idx, f"Unknown class {class_idx}")

st.title("Twitter Sentiment Analyzer")
text = st.text_input("Enter a word:")

if text:
    st.success(f"Prediction: {predict_word(text)}")
