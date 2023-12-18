from datetime import datetime, timedelta
import time

from airflow import DAG

from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
# Initialize a count variable, currently unused in this script.
count = 0

# Define default arguments for the DAG (Directed Acyclic Graph). These settings include:
# - The owner of the DAG.
# - A flag indicating whether the DAG's execution depends on the past runs.
# - Email settings for notifications.
# - Retry policy in case of failure.
default_args = {
    'owner': 'sf3209',
    'depends_on_past': False,
    'email': ['sf3209@columbia.edu'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
    'queue': 'bash_queue',
    'pool': 'backfill'
}

# Create a new DAG instance with specific parameters:
# - The DAG's name ('STGCN').
# - The default arguments defined earlier.
# - A brief description of the DAG.
# - The schedule interval for the DAG's execution (once a day).
# - The start date for the DAG (set to the current time).
# - A flag to disable catchup for missed executions.
# - Tags for categorizing the DAG.
with DAG(
        'STGCN',
        default_args=default_args,
        description='STGCN on Stock Data',
        schedule_interval=timedelta(days=1),
        start_date=datetime.now(),
        catchup=False,
        tags=['project'],
) as dag:

    # Define three tasks using BashOperator, each executing a Python script:
    # - t1 runs 'stock_data.py'
    # - t2 runs 'weight_data.py'
    # - t3 runs 'train.py'
    t1 = BashOperator(
        task_id='Stock_Data',
        bash_command='python3 /Users/xx/airflow/dags/stock_data.py',
    )

    t2 = BashOperator(
        task_id='Weight_Data',
        bash_command='python3 /Users/xx/airflow/dags/weight_data.py',
    )

    t3 = BashOperator(
        task_id='Train_Eval',
        bash_command='python3 /Users/xx/airflow/dags/train.py',
    )

    # Define dependencies between tasks:
    # - 'Stock_Data' must complete before 'Weight_Data' starts.
    # - 'Weight_Data' must complete before 'Train_Eval' starts.
    t1 >> t2
    t2 >> t3









