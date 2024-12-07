import streamlit as st
from transformers import TextStreamer
from unsloth import FastLanguageModel
import torch

# Ensure we're using CPU
torch.set_default_device('cpu')

# Show title and description
st.title("👩‍🍳 Local Cooking Assistant (CPU)")
st.write(
    "This is a chatbot that uses a local fine-tuned model specialized in cooking and recipes. "
    "The model runs entirely on your CPU - no GPU required!"
)

class StreamlitTextStreamer(TextStreamer):
    def __init__(self, tokenizer, container, skip_prompt=True):
        super().__init__(tokenizer=tokenizer, skip_prompt=skip_prompt)
        self.container = container
        self.text = ""
        
    def put(self, text):
        self.text += text
        self.container.markdown(self.text)

# Load the model (only once)
@st.cache_resource
def load_model():
    max_seq_length = 2048
    model_name_or_path = "davnas/Italian_Cousine_1.2"
    
    # CPU-specific loading configuration
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_length,
        dtype=torch.float32,  # Use float32 for CPU
        load_in_4bit=False,   # Disable 4-bit quantization for CPU
    )
    
    # Enable inference optimizations
    FastLanguageModel.for_inference(model)
    
    # Ensure model is on CPU
    model = model.to('cpu')
    return model, tokenizer

# Load model with a loading indicator
with st.spinner("Loading model... Please wait... (This might take a while on CPU)"):
    try:
        model, tokenizer = load_model()
        st.success("Model loaded successfully on CPU!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Add a warning about CPU performance
st.sidebar.warning("Running on CPU - responses might be slower")

# Chat input
if prompt := st.chat_input("Ask me about cooking..."):
    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        # Prepare model input
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to('cpu')  # Ensure inputs are on CPU

        # Generate response with streaming
        with st.chat_message("assistant"):
            response_container = st.empty()
            streamer = StreamlitTextStreamer(
                tokenizer=tokenizer,
                container=response_container
            )
            
            # Add a progress indicator for CPU generation
            with st.spinner("Generating response... (CPU processing)"):
                model.generate(
                    input_ids=inputs,
                    streamer=streamer,
                    max_new_tokens=128,
                    use_cache=True,
                    temperature=1.2,     # Slightly reduced for CPU
                    min_p=0.1,
                )
            
            # Store the response
            st.session_state.messages.append({
                "role": "assistant",
                "content": streamer.text
            })
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Add a button to clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()

# Add CPU-specific settings to sidebar
st.sidebar.subheader("CPU Settings")
st.sidebar.markdown("""
- Using float32 precision
- 4-bit quantization disabled
- Cache enabled for better performance
""")