import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import sqlparse

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "onkolahmet/Qwen2-0.5B-Instruct-SQL-generator", 
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("onkolahmet/Qwen2-0.5B-Instruct-SQL-generator")

# # Few-shot examples to include in each prompt
# examples = [
#     {
#         "question": "Get the names and emails of customers who placed an order in the last 30 days.",
#         "sql": "SELECT name, email FROM customers WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);"
#     },
#     {
#         "question": "Find all employees with a salary greater than 50000.",
#         "sql": "SELECT * FROM employees WHERE salary > 50000;"
#     },
#     {
#         "question": "List all product names and their categories where the price is below 50.",
#         "sql": "SELECT name, category FROM products WHERE price < 50;"
#     },
#     {
#         "question": "How many users registered in the year 2022?",
#         "sql": "SELECT COUNT(*) FROM users WHERE YEAR(registration_date) = 2022;"
#     }
# ]

def generate_sql(question, context=None):
    # Construct prompt with few-shot examples and context if available
    prompt = "Translate natural language questions to SQL queries.\n\n"
    
    # Add table context if available
    if context and context.strip():
        prompt += f"Table Context:\n{context}\n\n"
    
    # # Add few-shot examples
    # for ex in examples:
    #     prompt += f"Q: {ex['question']}\nSQL: {ex['sql']}\n\n"
    
    # Add the current question
    prompt += f"Q: {question}\nSQL:"
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate SQL query
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=128,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Extract and decode only the new generation
    sql_query = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return sql_query.strip()

def clean_sql_output(sql_text):
    """
    Clean and deduplicate SQL queries:
    1. Remove comments
    2. Remove duplicate queries
    3. Extract only the most relevant query
    4. Format properly
    """
    # Remove SQL comments (both single line and multi-line)
    sql_text = re.sub(r'--.*?$', '', sql_text, flags=re.MULTILINE)
    sql_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.DOTALL)
    
    # Remove markdown code block syntax if present
    sql_text = re.sub(r'```sql|```', '', sql_text)
    
    # Split into individual queries if multiple exist
    if ';' in sql_text:
        queries = [q.strip() for q in sql_text.split(';') if q.strip()]
    else:
        # If no semicolons, try to identify separate queries by SELECT statements
        sql_text_cleaned = re.sub(r'\s+', ' ', sql_text)
        select_matches = list(re.finditer(r'SELECT\s+', sql_text_cleaned, re.IGNORECASE))
        
        if len(select_matches) > 1:
            queries = []
            for i in range(len(select_matches)):
                start = select_matches[i].start()
                end = select_matches[i+1].start() if i < len(select_matches) - 1 else len(sql_text_cleaned)
                queries.append(sql_text_cleaned[start:end].strip())
        else:
            queries = [sql_text]
    
    # Remove empty queries
    queries = [q for q in queries if q.strip()]
    
    if not queries:
        return ""
    
    # If we have multiple queries, need to deduplicate
    if len(queries) > 1:
        # Normalize queries for comparison (lowercase, remove extra spaces)
        normalized_queries = []
        for q in queries:
            # Use sqlparse to format and normalize
            try:
                formatted = sqlparse.format(
                    q + ('' if q.strip().endswith(';') else ';'), 
                    keyword_case='lower',
                    identifier_case='lower', 
                    strip_comments=True,
                    reindent=True
                )
                normalized_queries.append(formatted)
            except:
                # If sqlparse fails, just do basic normalization
                normalized = re.sub(r'\s+', ' ', q.lower().strip())
                normalized_queries.append(normalized)
        
        # Find unique queries
        unique_queries = []
        unique_normalized = []
        
        for i, norm_q in enumerate(normalized_queries):
            if norm_q not in unique_normalized:
                unique_normalized.append(norm_q)
                unique_queries.append(queries[i])
        
        # Choose the most likely correct query:
        # 1. Prefer queries with SELECT
        # 2. Prefer longer queries (often more detailed)
        # 3. Prefer first query if all else equal
        select_queries = [q for q in unique_queries if re.search(r'SELECT\s+', q, re.IGNORECASE)]
        
        if select_queries:
            # Choose the longest SELECT query (likely most detailed)
            best_query = max(select_queries, key=len)
        elif unique_queries:
            # If no SELECT queries, choose the longest query
            best_query = max(unique_queries, key=len)
        else:
            # Fallback to the first query
            best_query = queries[0]
    else:
        best_query = queries[0]
    
    # Clean up the chosen query
    best_query = best_query.strip()
    if not best_query.endswith(';'):
        best_query += ';'
    
    # Final formatting to ensure consistent spacing
    best_query = re.sub(r'\s+', ' ', best_query)
    
    try:
        # Use sqlparse to nicely format the SQL for display
        formatted_sql = sqlparse.format(
            best_query,
            keyword_case='upper',
            identifier_case='lower',
            reindent=True,
            indent_width=2
        )
        return formatted_sql
    except:
        return best_query

def process_input(question, table_context):
    """Function to process user input through the model and return formatted results"""
    if not question.strip():
        return "Please enter a question."
    
    # Generate SQL from the question and context
    raw_sql = generate_sql(question, table_context)
    
    # Clean the SQL output
    cleaned_sql = clean_sql_output(raw_sql)
    
    if not cleaned_sql:
        return "Sorry, I couldn't generate a valid SQL query. Please try rephrasing your question."
    
    return cleaned_sql

# Sample table context examples for the example selector
example_contexts = [
    # Example 1
    """
CREATE TABLE customers (
  id INT PRIMARY KEY,
  name VARCHAR(100),
  email VARCHAR(100),
  order_date DATE
);
    """,
    
    # Example 2
    """
CREATE TABLE products (
  id INT PRIMARY KEY,
  name VARCHAR(100),
  category VARCHAR(50),
  price DECIMAL(10,2),
  stock_quantity INT
);
    """,
    
    # Example 3
    """
CREATE TABLE employees (
  id INT PRIMARY KEY,
  name VARCHAR(100),
  department VARCHAR(50),
  salary DECIMAL(10,2),
  hire_date DATE
);
CREATE TABLE departments (
  id INT PRIMARY KEY,
  name VARCHAR(50),
  manager_id INT,
  budget DECIMAL(15,2)
);
    """
]

# Sample question examples
example_questions = [
    "Get the names and emails of customers who placed an order in the last 30 days.",
    "Find all products with less than 10 items in stock.",
    "List all employees in the Sales department with a salary greater than 50000.",
    "What is the total budget for departments with more than 5 employees?",
    "Count how many products are in each category where the price is greater than 100."
]

# Create the Gradio interface
with gr.Blocks(title="Text to SQL Converter") as demo:
    gr.Markdown("# Text to SQL Query Converter")
    gr.Markdown("Enter your question and optional table context to generate an SQL query.")
    
    

# Launch the app
demo.launch()