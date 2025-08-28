import streamlit as st

def main():
    st.title("Energy Management RL Dashboard Test")
    st.write("If you can see this, Streamlit is working correctly!")
    
    st.write("This is a simple test to verify that Streamlit is installed and functioning properly.")
    
    st.header("Next Steps")
    st.write("1. Close this test page")
    st.write("2. Run the full dashboard with: `streamlit run dashboard/app.py`")
    
    if st.button("Click me!"):
        st.success("Button clicked! Streamlit interactivity is working.")

if __name__ == "__main__":
    main()
