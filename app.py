import streamlit as st

# Load the model and tokenizer
model_path = 'bert_model.pkl'
tokenizer_path = 'bert_tokenizer.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

model.eval()

# Function to predict sentiment
def predict_sentiment(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)
    return predicted_labels.cpu().numpy(), probabilities.cpu().numpy()

# Streamlit interface
st.title("Firm Reputation Classifier")
firm_name = st.text_input("Enter the firm name:")
if st.button("Predict"):
    firm_reviews = df[df['firm'] == firm_name]['headline'].tolist()
    if len(firm_reviews) == 0:
        st.write("No reviews found for this firm.")
    else:
        predicted_labels, probabilities = predict_sentiment(firm_reviews)
        avg_sentiment = np.mean(predicted_labels)
        sentiment = 'good' if avg_sentiment > 0.5 else 'bad'
        st.write(f"The firm is predicted to be: {sentiment}")
        
        # LIME explanation
        explainer = lime.lime_text.LimeTextExplainer(class_names=['bad', 'good'])
        exp = explainer.explain_instance(firm_reviews[0], lambda x: predict_sentiment(x)[1], num_features=10)
        exp.show_in_notebook(text=True)
        st.write("LIME Explanation")
        components.html(exp.as_html(), height=800)
