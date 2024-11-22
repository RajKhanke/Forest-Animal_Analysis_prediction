import streamlit as st
from tigers import tigers_page
from leopard import leopard_page
from elephants import elephants_page  # Ensure this function exists in elephants.py

# Set up the page configuration
st.set_page_config(page_title="Animal Monitoring System", layout="wide")

# Create a main menu with a selection box
st.sidebar.title("Forest and Wildlife Framework")

# Main selection box for system type
main_selection = st.sidebar.selectbox(
    "Select System",
    ["Animal Monitoring System"]
)

# Conditional display based on the main selection
if main_selection == "Animal Monitoring System":
    # Show the sidebar for animal monitoring options
    st.sidebar.title("Animal Monitoring System")

    # Sidebar menu for animal types
    animal_selection = st.sidebar.selectbox(
        "Go to",
        ["Tigers", "Leopards", "Elephants"]
    )

    # Display the selected animal monitoring page
    if animal_selection == "Tigers":
        tigers_page()
    elif animal_selection == "Leopards":
        leopard_page()
    elif animal_selection == "Elephants":
        elephants_page()  # Ensure you have elephants_page function in elephants.py


