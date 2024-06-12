import streamlit as st
from mlx_lm import load, generate, stream_generate
from mlx_lm.utils import generate_step
import time

st.set_page_config(page_title="Local MLX Chatbot", page_icon=":robot_face:", layout="wide")

# Cache the model loading to avoid reloading every time
@st.cache_resource
def load_model():
    MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
    ADAPTER = None

    if ADAPTER:
        model, tokenizer = load(path_or_hf_repo=MODEL, adapter_path=ADAPTER)
    else:
        model, tokenizer = load(path_or_hf_repo=MODEL)

    return model, tokenizer

model, tokenizer = load_model()

def remove_tag(response_generator):
    tag_start = "<|start_header_id|>assistant<|end_header_id|>"
    tag_buffer = ""
    
    for token in response_generator:
        tag_buffer += token
        
        if tag_start in tag_buffer:
            tag_buffer = tag_buffer[len(tag_start):]
        
        if len(tag_buffer) > len(tag_start):
            yield tag_buffer[:len(tag_buffer) - len(tag_start)]
            tag_buffer = tag_buffer[-len(tag_start):]
    
    if tag_buffer:
        yield tag_buffer

def generate_response_streaming(prompt: str, temp: float, max_tokens: int):
    response_generator = stream_generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=max_tokens, 
        temp=temp
    )
    
    # for token in response_generator:
    #     yield token
    for token in remove_tag(response_generator):
        yield token

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.sidebar.title("Sidebar")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

if clear_button:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)
        

# Chat input
prompt = st.chat_input("You:")
if prompt:
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    formatted_conversation = tokenizer.apply_chat_template(st.session_state['messages'], tokenize=False)
    tic = time.time()
    
    # Create a placeholder for the assistant's message
    assistant_message = st.empty()
    full_response = ""
    token_count = 0

    for token in generate_response_streaming(formatted_conversation, temp=0.0, max_tokens=1000):
        full_response += token
        token_count += 1
        assistant_message.markdown(full_response)
        
    st.session_state['messages'].append({"role": "assistant", "content": full_response})

    tokens = tokenizer.tokenize(full_response)
    num_tokens_generated = len(tokens)
    generation_time = time.time() - tic
    generation_tps = num_tokens_generated / generation_time
    tokens = tokenizer.tokenize(formatted_conversation)
    num_tokens_total = len(tokens) + num_tokens_generated
    
    st.markdown("---")
    st.markdown(f"<small><font color='gray'>**Number of tokens generated:** {num_tokens_generated} --- **Time:** {generation_time:.1f} seconds --- **TPS:** {generation_tps:.1f}</font></small>", unsafe_allow_html=True)
    st.markdown(f"<small><font color='gray'>**Number of total tokens in conversation:** {num_tokens_total}</font></small>", unsafe_allow_html=True)

# Hide streamlit style
hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)