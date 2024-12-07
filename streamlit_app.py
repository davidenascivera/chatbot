import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Ensure we're only using CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_default_device('cpu')

# Show title and description
st.title("👩‍🍳 Local Cooking Assistant (CPU)")
st.write(
    "This is a chatbot that uses a local fine-tuned model specialized in cooking and recipes. "
    "The model runs entirely on your CPU - no GPU required!"
)

# Custom streamer for Streamlit
class SimpleStreamlitStreamer:
    def __init__(self, tokenizer, container):
        self.tokenizer = tokenizer
        self.container = container
        self.text = ""
        
    def put(self, token_ids):
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        self.text += text
        self.container.markdown(self.text)
        
    def end(self):
        pass

# Load the model (only once)
@st.cache_resource
def load_model():
    model_name_or_path = "davnas/Italian_Cousine_1.2"
    
    # CPU-specific loading configuration
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=None,  # Don't use device map
        load_in_8bit=False,  # Don't use 8-bit quantization
        load_in_4bit=False,  # Don't use 4-bit quantization
        torch_dtype=torch.float32,  # Use standard precision
        low_cpu_mem_usage=True
    )
    
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
        )

        # Generate response with streaming
        with st.chat_message("assistant"):
            response_container = st.empty()
            streamer = SimpleStreamlitStreamer(
                tokenizer=tokenizer,
                container=response_container
            )
            
            # Add a progress indicator for CPU generation
            with st.spinner("Generating response... (CPU processing)"):
                generated_ids = model.generate(
                    inputs,
                    max_length=256,
                    do_sample=True,
                    temperature=1.2,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Decode the complete response
                response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                response_container.markdown(response)
            
            # Store the response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Add a button to clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()

# Add CPU-specific settings to sidebar
st.sidebar.subheader("Settings")
st.sidebar.markdown("""
- Running on CPU only
- Using standard 32-bit precision
- No quantization
- Low memory usage enabled
""")