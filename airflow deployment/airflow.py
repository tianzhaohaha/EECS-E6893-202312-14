from datetime import datetime, timedelta
import time

from airflow import DAG

from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


count = 0


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

with DAG(
        'STGCN',
        default_args=default_args,
        description='STGCN on Stock Data',
        schedule_interval=timedelta(days=1),
        start_date=datetime.now(),
        catchup=False,
        tags=['project'],
) as dag:

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

    t1 >> t2
    t2 >> t3









