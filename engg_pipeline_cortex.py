import streamlit as st
import pandas as pd
import snowflake.connector
from tabulate import tabulate
import io

st.set_page_config(page_title="Cortex AI-Powered Snowflake ETL Chatbot", layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = 0
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'db_connection' not in st.session_state:
    st.session_state.db_connection = None
if 'db_params' not in st.session_state:
    st.session_state.db_params = {}
if 'transformed_df' not in st.session_state:
    st.session_state.transformed_df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'etl_logic' not in st.session_state:
    st.session_state.etl_logic = ""

st.sidebar.title("📌 Navigation")
st.sidebar.markdown("---")
pages = {
    "🏠 Home": 0,
    "📂 Data Upload & ETL": 1,
    "🔗 Snowflake Connection": 2,
    "🤖 AI Chat & SQL Queries": 3
}
selected_page = st.sidebar.radio("Go to", list(pages.keys()), index=st.session_state.page, key="sidebar_nav")
st.session_state.page = pages[selected_page]
st.sidebar.markdown(f"**🔹 You are on:** {selected_page}")
st.sidebar.markdown("---")

def get_available_models():
    conn = st.session_state.get("db_connection")
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute("SHOW MODELS IN SNOWFLAKE.CORTEX")
        models = [row[0] for row in cursor.fetchall()]
        return models
    except Exception as e:
        return []

def llm_markdown_to_dataframe(markdown_text):
    """Converts an LLM-generated Markdown table into a Pandas DataFrame."""
    try:
        # Read the Markdown table as CSV with a pipe `|` separator
        df = pd.read_csv(io.StringIO(markdown_text), sep="|", skipinitialspace=True)

        # Strip whitespace from all columns
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # Drop empty first and last columns (artifacts from Markdown formatting)
        df = df.iloc[:, 1:-1]

        # Rename columns using first row, then drop it
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

        return df
    except Exception as e:
        print(f"❌ Error processing Markdown table: {e}")
        return pd.DataFrame()  # Return empty DataFrame if conversion fails
    
import pandas as pd
import streamlit as st

def llm_table_to_csv(llm_output: str):
    """
    Converts an LLM-generated table (text format) into a Pandas DataFrame and saves it as a CSV file.

    Parameters:
    - llm_output (str): The LLM output table as a string.
    - csv_filename (str): The name of the CSV file to save.

    Returns:
    - df (pd.DataFrame): The cleaned DataFrame.
    - csv_data (bytes): CSV content in bytes for Streamlit download.
    """

    # Step 1: Convert Multiline String to List of Lines
    lines = llm_output.strip().split("\n")

    # Step 2: Identify Header and Data Lines
    header = []
    data = []

    for line in lines:
        # Ignore separator lines (like "|---:|")
        if "---" in line:
            continue

        # Step 3: Split each line using "|"
        values = line.split("|")[1:-1]  # Remove empty first and last values

        # Step 4: Trim spaces
        values = [v.strip() for v in values]

        # Step 5: Append header separately
        if not header:
            header = values  # First valid row is the header
        else:
            data.append(values)

    # Step 6: Convert Data to Pandas DataFrame
    df = pd.DataFrame(data, columns=header)

    return df

def query_snowflake_llm(prompt, df=None):
    conn = st.session_state.get("db_connection")
    if not conn:
        return "❌ Not connected to Snowflake. Please log in."
    try:
        cursor = conn.cursor()
        model = st.session_state.selected_model if st.session_state.selected_model else 'llama3.1-405b'
        # llama3.1-405b
        # mistral-large2
        if df is not None:
            df_sample = tabulate(df, headers='keys', tablefmt='pipe')
            # df_sample=df.to_markdown(index=False)
            query = "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s)"
            cursor.execute(query, (model, f'Prompt: {prompt}\n\nData Sample:\n{df_sample}'))
        else:
            query = "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s)"
            cursor.execute(query, (model, prompt))
        result = cursor.fetchone()
        return result[0] if result else "❌ No response from Snowflake LLM."
    except Exception as e:
        return f"❌ Error: {e}"

if st.session_state.page == 1:
    st.title("📂 Data Upload & ETL")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_file = df
        st.success("✅ File uploaded successfully!")
    else:
        st.warning("⚠️ Please upload a CSV file to proceed.")
        df = None
    
    if df is not None:
        st.subheader("🔍 Preview Dataset")
        st.dataframe(df.head())
    
    
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous"):
            st.session_state.page = 0
            st.rerun()
    with col2:
        if df is not None and st.button("Next →"):
            st.session_state.page = 2
            st.rerun()

if st.session_state.page == 2:
    st.title("🔗 Connect to Snowflake")
    connection_params = {
        "account": st.text_input("Account Identifier"),
        "user": st.text_input("Username"),
        "password": st.text_input("Password", type="password"),
        "database": st.text_input("Database Name"),
        "schema": st.text_input("Schema"),
        "warehouse": st.text_input("Warehouse"),
    }
    
    def test_connection():
        try:
            conn = snowflake.connector.connect(**connection_params)
            st.session_state.db_connection = conn
            st.success("✅ Connected to Snowflake successfully!")
            st.session_state.available_models = get_available_models()
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")
    
    if st.button("Test Connection"):
        test_connection()
    
    if 'available_models' in st.session_state and st.session_state.available_models:
        st.session_state.selected_model = st.selectbox("Select Snowflake Cortex Model", st.session_state.available_models)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous"):
            st.session_state.page = 1
            st.rerun()
    with col2:
        if st.button("Next →"):
            st.session_state.page = 3
            st.rerun()

if st.session_state.page == 3:
    st.session_state.available_models = get_available_models()
    if 'available_models' in st.session_state and st.session_state.available_models:
        st.session_state.selected_model = st.selectbox("Select Snowflake Cortex Model", st.session_state.available_models)
    st.title("🤖 AI Chat for ETL & SQL Queries")
    # st.subheader("🛠️ Define ETL Logic")
    # st.session_state.etl_logic = st.text_area("Enter your custom ETL transformation logic")
    st.subheader("📂 Uploaded Dataset")
    st.dataframe(st.session_state.uploaded_file)
    
    if st.session_state.uploaded_file is None:
        st.warning("⚠️ No file uploaded. Please go back and upload a dataset.")
        st.stop()
    df = st.session_state.uploaded_file
    
    user_input = st.chat_input("Type your message...")
    if user_input:
        try:
            # Query Snowflake LLM and get a cleaned DataFrame
            response = query_snowflake_llm(user_input + " and Output only the final cleaned table, with no additional text or explanation.", df)

            # Ensure response is a DataFrame
            if not isinstance(response, pd.DataFrame):
                st.error("❌ The response is not a DataFrame. Please check the output.")
            else:
                # Update chat history
                st.session_state.chat_history.append(("🧑", user_input))
                st.session_state.chat_history.append(("🤖", "✅ Dataframe generated successfully."))

                # Display chat history
                for role, message in st.session_state.chat_history:
                    if role == "🧑":
                        st.chat_message("user").write(message)
                    else:
                        st.chat_message("assistant").write(message)

                # Proceed with Snowflake upload
                conn = st.session_state.get("db_connection")
                if conn:
                    cursor = conn.cursor()

                    # Get table name from user
                    table_name = st.text_input("Enter table name for cleaned data", "cleaned_dataset")

                    if st.button("Upload to Snowflake"):
                        try:
                            # Ensure column names are safe for SQL
                            columns = ', '.join([f'"{col}" STRING' for col in response.columns])

                            # Create table in Snowflake
                            create_table_query = f"CREATE OR REPLACE TABLE {table_name} ({columns})"
                            cursor.execute(create_table_query)

                            # Prepare data for bulk insert
                            rows = response.values.tolist()  # Handle NaN values

                            # Generate parameterized query for bulk insert
                            placeholders = ", ".join(["%s"] * len(response.columns))
                            insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"

                            # Execute batch insert
                            cursor.executemany(insert_query, rows)

                            st.success(f"✅ Data uploaded successfully to Snowflake table: {table_name}")

                        except Exception as e:
                            st.error(f"❌ Upload failed: {e}")

                else:
                    st.error("❌ No Snowflake connection found. Please connect first.")

        except Exception as e:
            st.error(f"❌ Query failed: {e}")
    
    st.subheader("🔧 ETL Operations")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clean Data"):
            response = query_snowflake_llm("Clean the dataset by handling missing values accordingly and removing duplicates. Output only the final cleaned table, with no additional text or explanation.", df)
            st.session_state.chat_history.append(("🤖", response))
            st.chat_message("assistant").write(response)
            
            if response:
                try:
                    conn = st.session_state.get("db_connection")
                    if conn:
                        cursor = conn.cursor()
                        table_name = st.text_input("Enter table name for cleaned data", "cleaned_dataset")
                        if st.button("Upload to Snowflake"):
                            cursor.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM VALUES {','.join(str(tuple(row)) for row in df.values)}")
                            st.success(f"✅ Data uploaded to Snowflake table: {table_name}")
                    else:
                        st.error("❌ No Snowflake connection found. Please connect first.")
                except Exception as e:
                    st.error(f"❌ Upload failed: {e}")
            response = query_snowflake_llm("Clean the dataset, handle missing values, and remove duplicates and return only table nothing else", df)
            st.session_state.chat_history.append(("🤖", response))
            st.chat_message("assistant").write(response)
    with col2:
        if st.button("📊 Summarize Data"):
            response = query_snowflake_llm("Summarize the key insights from the dataset.", df)
            st.session_state.chat_history.append(("🤖", response))
            st.chat_message("assistant").write(response)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous"):
            st.session_state.page = 2
            st.rerun()
    with col2:
        if st.button("Restart 🔄"):
            st.session_state.page = 0
            st.session_state.uploaded_file = None
            st.session_state.db_connection = None
            st.session_state.chat_history = []
            st.rerun()
