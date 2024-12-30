from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Function to be run by the PythonOperator
def hello_world():
    print("Hello, World!")

# Define the DAG
dag = DAG(
    'hello_world', 
    description='A simple Hello World DAG',
    schedule_interval=None,  # No automatic scheduling
    start_date=datetime(2024, 12, 26), 
    catchup=False,  # Do not backfill the DAG runs
)

# Create a task that runs the hello_world function
hello_task = PythonOperator(
    task_id='hello_task',
    python_callable=hello_world,
    dag=dag,
)

# If there were multiple tasks, you could set task dependencies here
hello_task
