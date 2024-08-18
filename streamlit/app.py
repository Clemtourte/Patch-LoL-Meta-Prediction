import subprocess
import sys
import os
import streamlit as st

# Streamlit app to collect user information
st.title("User Information")

region = st.selectbox("Select your region (only EUW available for now):", ["EUW1"])
username = st.text_input("Enter your username:")
tag = st.text_input("Enter your tag:")

if st.button("Search"):
    # Get the path to the Python interpreter in the current environment
    python_interpreter = sys.executable

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    # Optionally, run the main.py script with the correct relative path
    result = subprocess.run([python_interpreter, os.path.join("..", "src", "main.py"), username, tag, region], capture_output=True, text=True, env=env)
    
    if result.returncode == 0:
        st.success("User information saved and main.py executed.")
    else:
        st.error(f"Error executing main.py: {result.stderr}")