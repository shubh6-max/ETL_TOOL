# Cortex AI-Powered Snowflake ETL Chatbot

## Overview
The **Cortex AI-Powered Snowflake ETL Chatbot** is an interactive web-based application built with **Streamlit** that enables users to:

- Upload and clean CSV data.
- Connect to a Snowflake database.
- Query Snowflake Cortex AI models for data transformation and insights.
- Store processed data into Snowflake tables.
- Perform ETL operations using AI-generated transformations.

## Features
### 📂 Data Upload & ETL
- Users can upload CSV files for processing.
- Data preview is displayed within the application.

### 🔗 Snowflake Connection
- Securely connect to a Snowflake database using user-provided credentials.
- Retrieve available **Cortex AI** models from the connected Snowflake instance.

### 🤖 AI Chat & SQL Queries
- Users can interact with **Snowflake Cortex AI** models via a chatbot interface.
- AI-generated transformations and queries are applied directly to uploaded data.
- Results are displayed and can be uploaded to Snowflake.

### 🛠️ ETL Operations
- **Data Cleaning:** Automatically handles missing values and removes duplicates.
- **Data Summarization:** Generates key insights from the dataset.
- **Data Upload:** Allows users to specify a table name and store the transformed data in Snowflake.

## Installation
### Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **pip (Python package manager)**
- **Snowflake Connector for Python**
- **Streamlit**
- **pandas**
- **tabulate**

### Installation Steps
1. Clone this repository:
   ```sh
   git clone [https://github.com/your-repo/cortex-ai-snowflake-etl-chatbot.git](https://github.com/shubh6-max/ETL_TOOL.git)
   cd cortex-ai-snowflake-etl-chatbot
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the application:
   ```sh
   streamlit run app.py
   ```

## Usage
1. **Upload a CSV File**
   - Navigate to the **Data Upload & ETL** page.
   - Upload a CSV file.
   - Preview the data.
2. **Connect to Snowflake**
   - Enter your **Snowflake Account Identifier, Username, Password, Database, Schema, and Warehouse**.
   - Click **Test Connection** to verify access.
   - Select an available **Cortex AI model**.
3. **Chat with AI for Data Processing**
   - Enter queries to transform the dataset.
   - AI processes the data and returns cleaned results.
4. **Upload Cleaned Data to Snowflake**
   - Enter a table name.
   - Click **Upload to Snowflake** to store the transformed data.
5. **Perform ETL Operations**
   - **Clean Data:** Handle missing values and remove duplicates.
   - **Summarize Data:** Generate key insights from the dataset.
   
## Error Handling & Debugging
- Ensure that **Snowflake credentials** are correctly entered before attempting a connection.
- If the AI response is not a valid DataFrame, verify the input data format.
- Check the **chat history** for any errors in the query response.

## Technologies Used
- **Python**
- **Streamlit**
- **Pandas**
- **Snowflake Connector for Python**
- **Snowflake Cortex AI**

## Future Enhancements
- Implement **authentication** for Snowflake connections.
- Improve **error logging** for debugging.
- Add support for **multi-step ETL pipelines**.
- Provide more **AI-driven transformations**.

## License
This project is licensed under the MIT License.

---
Developed with ❤️ for seamless AI-powered ETL with Snowflake Cortex AI 🚀.
