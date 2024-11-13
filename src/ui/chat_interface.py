# src/ui/chat_interface.py
import streamlit as st
import time
from typing import Tuple, Optional
from datetime import datetime
from ..core.types import MessageRole
from ..inference.mlx_engine import MLXInference
from ..config.model_config import ModelConfig


class ChatUI:
    """Streamlit UI for the chat interface."""

    def __init__(self):
        """Initialize the Chat UI with custom styling."""
        self.inference_engine = self.get_inference_engine()
        self.init_session_state()
        self.apply_custom_css()

    @staticmethod
    def apply_custom_css():
        """Apply minimal CSS styling that only hides Streamlit elements."""
        st.markdown("""
            <style>
            /* Hide Streamlit elements */
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

            /* Minimal styling for sidebar and custom elements */
            .status-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 5px;
            }
            
            .status-online {
                background-color: #4CAF50;
            }
            
            .status-offline {
                background-color: #F44336;
            }

            /* Message timestamp */
            .message-timestamp {
                font-size: 0.8em;
                color: #666;
                margin-top: 0.2rem;
            }

            /* System message */
            .system-message {
                padding: 0.5rem 1rem;
                border-radius: 5px;
                margin: 0.5rem 0;
                font-size: 0.9em;
                color: #E65100;
            }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    @st.cache_resource
    def get_inference_engine() -> MLXInference:
        """Cache the inference engine instance."""
        return MLXInference()

    def init_session_state(self):
        """Initialize session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "max_history" not in st.session_state:
            st.session_state.max_history = ModelConfig.DEFAULT_MAX_HISTORY
        if "auto_clear" not in st.session_state:
            st.session_state.auto_clear = False
        if "last_response_time" not in st.session_state:
            st.session_state.last_response_time = None
        

    def format_message_for_display(self, role: str, content: str, timestamp: Optional[float] = None) -> str:
        """Format message with optional timestamp and styling."""
        formatted_message = content
        
        if st.session_state.show_timestamps and timestamp:
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            formatted_message += f"\n<div class='message-timestamp'>{time_str}</div>"
        
        return formatted_message

    def display_chat_messages(self):
        """Display chat messages."""
        if not st.session_state.messages:
            return

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def show_conversation_stats(self):
        """Display conversation statistics."""
        if st.session_state.messages:
            st.sidebar.subheader("Conversation Stats")
            
            # Calculate stats
            total_messages = len(st.session_state.messages)
            user_messages = sum(1 for m in st.session_state.messages if m["role"] == MessageRole.USER)
            assistant_messages = sum(1 for m in st.session_state.messages if m["role"] == MessageRole.ASSISTANT)
            
            # Duration
            duration = datetime.now() - st.session_state.conversation_started
            duration_minutes = duration.total_seconds() / 60
            
            # Display stats
            col1, col2 = st.sidebar.columns(2)
            col1.metric("Total Messages", total_messages)
            col2.metric("Duration", f"{duration_minutes:.1f}m")
            
            col3, col4 = st.sidebar.columns(2)
            col3.metric("Your Messages", user_messages)
            col4.metric("Bot Messages", assistant_messages)

    def show_sidebar(self):
        """Display sidebar with settings and status."""
        with st.sidebar:
            st.title("Settings")
            
            # Server Status
            is_healthy, message = self.inference_engine.check_server_health()
            status_color = "status-online" if is_healthy else "status-offline"
            st.markdown(f"""
                <div>
                    <span class="status-indicator {status_color}"></span>
                    <span>{message}</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Chat Settings
            st.subheader("Chat Settings")
            
            # Model settings
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=1.0,
                value=ModelConfig.DEFAULT_TEMPERATURE,
                step=0.1,
                help="Higher values make the output more random"
            )
            
            max_tokens = st.slider(
                "Max Response Length",
                min_value=100,
                max_value=ModelConfig.DEFAULT_MAX_TOKENS,
                value=ModelConfig.DEFAULT_MAX_TOKENS,
                step=100,
                help="Maximum number of tokens in the response"
            )
            
            # History settings
            st.session_state.max_history = st.slider(
                "Max Chat History",
                min_value=10,
                max_value=100,
                value=ModelConfig.DEFAULT_MAX_HISTORY,
                help="Maximum number of messages to keep in history"
            )
            
            # Actions
            st.subheader("Actions")
            if st.button("Clear History", type="primary"):
                st.session_state.messages = []
                st.experimental_rerun()

    def export_chat_history(self):
        """Export chat history as JSON."""
        if not st.session_state.messages:
            st.sidebar.warning("No messages to export!")
            return
            
        chat_export = {
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages,
            "stats": {
                "total_messages": len(st.session_state.messages),
                "duration_minutes": (datetime.now() - st.session_state.conversation_started).total_seconds() / 60
            }
        }
        
        st.sidebar.download_button(
            label="Download JSON",
            data=str(chat_export),
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    def process_user_input(self):
        """Process user input with response time calculation."""
        if prompt := st.chat_input("What's on your mind?"):
            is_valid, error_message = self.validate_input(prompt)
            if not is_valid:
                st.warning(error_message)
                return

            # Auto-clear if enabled and history limit reached
            if len(st.session_state.messages) >= st.session_state.max_history:
                st.session_state.messages.pop(0)  # Remove oldest message

            # Add user message
            st.session_state.messages.append({
                "role": MessageRole.USER,
                "content": prompt
            })

            with st.chat_message(MessageRole.USER):
                st.markdown(prompt)

            # Process response
            with st.chat_message(MessageRole.ASSISTANT):
                response_placeholder = st.empty()
                full_response = ""

                try:
                    response_start_time = time.time()
                    st.session_state.last_response_time = response_start_time

                    with st.spinner("Thinking..."):
                        message_list = [
                            {"role": m["role"], "content": m["content"]} 
                            for m in st.session_state.messages
                        ]

                        for chunk in self.inference_engine.chat_completion(
                            message_list,
                            max_tokens=st.session_state.get("max_tokens", ModelConfig.DEFAULT_MAX_TOKENS),
                            temperature=st.session_state.get("temperature", ModelConfig.DEFAULT_TEMPERATURE)
                        ):
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)

                    # Calculate and display response time
                    response_time = time.time() - response_start_time
                    st.caption(f"Response time: {response_time:.2f}s")

                    # Add to history
                    st.session_state.messages.append({
                        "role": MessageRole.ASSISTANT,
                        "content": full_response
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.messages.pop()  # Remove failed message
                    
    def validate_input(self, prompt: str) -> Tuple[bool, Optional[str]]:
        """Validate user input."""
        if not prompt.strip():
            return False, "Please enter a non-empty message"
        return True, None
    
    def calculate_response_time(self) -> str:
        """Calculate and display response time."""
        if st.session_state.last_response_time:
            response_time = time.time() - st.session_state.last_response_time
            return f"Response time: {response_time:.2f} seconds"
        return ""
