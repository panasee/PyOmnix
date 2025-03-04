"""
PyOmnix Workflow Framework Example

This example demonstrates how to use the PyOmnix workflow framework to create
and execute workflows in your Python applications.
"""

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from PyOmnix.airflow_interface import create_workflow, TaskDecorator


# Create a workflow
workflow = create_workflow(
    dag_id="data_processing_workflow",
    description="A workflow for data processing and analysis",
    schedule_interval=datetime.timedelta(days=1)
)

# Define task functions
def generate_data(ti, rows=100, **kwargs):
    """Generate random data for analysis."""
    print("Generating random data...")
    
    # Create a DataFrame with random data
    data = {
        'id': range(1, rows + 1),
        'value_a': np.random.normal(0, 1, rows),
        'value_b': np.random.normal(5, 2, rows),
        'category': np.random.choice(['A', 'B', 'C'], rows)
    }
    df = pd.DataFrame(data)
    
    # Save the data to a temporary file
    output_dir = Path("temp_data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "random_data.csv"
    df.to_csv(output_path, index=False)
    
    return {
        "data_path": str(output_path),
        "rows": rows
    }


def process_data(ti, data_path, **kwargs):
    """Process the generated data."""
    print(f"Processing data from {data_path}...")
    
    # Read the data
    df = pd.read_csv(data_path)
    
    # Perform some processing
    df['value_sum'] = df['value_a'] + df['value_b']
    df['value_product'] = df['value_a'] * df['value_b']
    
    # Calculate statistics by category
    stats = df.groupby('category').agg({
        'value_a': ['mean', 'std'],
        'value_b': ['mean', 'std'],
        'value_sum': ['mean', 'std'],
        'value_product': ['mean', 'std']
    })
    
    # Save the processed data
    output_dir = Path("temp_data")
    processed_path = output_dir / "processed_data.csv"
    stats_path = output_dir / "stats_data.csv"
    
    df.to_csv(processed_path, index=False)
    stats.to_csv(stats_path)
    
    return {
        "processed_path": str(processed_path),
        "stats_path": str(stats_path)
    }


def visualize_data(ti, processed_path, stats_path, **kwargs):
    """Create visualizations from the processed data."""
    print(f"Creating visualizations from {processed_path}...")
    
    # Read the data
    df = pd.read_csv(processed_path)
    stats = pd.read_csv(stats_path)
    
    # Create output directory for plots
    output_dir = Path("temp_data/plots")
    output_dir.mkdir(exist_ok=True)
    
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    for category in df['category'].unique():
        subset = df[df['category'] == category]
        plt.scatter(subset['value_a'], subset['value_b'], label=f'Category {category}')
    
    plt.title('Value A vs Value B by Category')
    plt.xlabel('Value A')
    plt.ylabel('Value B')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    scatter_path = output_dir / "scatter_plot.png"
    plt.savefig(scatter_path)
    plt.close()
    
    # Create a histogram
    plt.figure(figsize=(12, 6))
    plt.hist(df['value_sum'], bins=20, alpha=0.5, label='Sum')
    plt.hist(df['value_product'], bins=20, alpha=0.5, label='Product')
    plt.title('Distribution of Sum and Product')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    hist_path = output_dir / "histogram.png"
    plt.savefig(hist_path)
    plt.close()
    
    return {
        "scatter_plot": str(scatter_path),
        "histogram": str(hist_path)
    }


def send_report(ti, scatter_plot, histogram, **kwargs):
    """Simulate sending a report with the visualizations."""
    print("Sending report...")
    print(f"Scatter plot: {scatter_plot}")
    print(f"Histogram: {histogram}")
    
    # In a real application, you might email these plots or save them to a report
    
    return {
        "report_status": "sent",
        "timestamp": datetime.datetime.now().isoformat()
    }


# Add tasks to the workflow
# Method 1: Using the add_task method
task_generate = workflow.add_task(
    task_id="generate_data",
    python_callable=generate_data,
    kwargs={"rows": 200}
)

task_process = workflow.add_task(
    task_id="process_data",
    python_callable=process_data,
    kwargs={"data_path": "{{ ti.xcom_pull(task_ids='generate_data')['data_path'] }}"}
)

# Method 2: Using the TaskDecorator
@TaskDecorator.task(
    workflow_manager=workflow,
    task_id="visualize_data",
    op_kwargs={
        "processed_path": "{{ ti.xcom_pull(task_ids='process_data')['processed_path'] }}",
        "stats_path": "{{ ti.xcom_pull(task_ids='process_data')['stats_path'] }}"
    }
)
def task_visualize(ti, processed_path, stats_path, **kwargs):
    return visualize_data(ti, processed_path, stats_path, **kwargs)

@TaskDecorator.task(
    workflow_manager=workflow,
    task_id="send_report",
    op_kwargs={
        "scatter_plot": "{{ ti.xcom_pull(task_ids='visualize_data')['scatter_plot'] }}",
        "histogram": "{{ ti.xcom_pull(task_ids='visualize_data')['histogram'] }}"
    }
)
def task_report(ti, scatter_plot, histogram, **kwargs):
    return send_report(ti, scatter_plot, histogram, **kwargs)

# Set task dependencies
task_generate >> task_process >> task_visualize
task_visualize >> task_report

# Alternative way to set dependencies
# workflow.set_dependencies("process_data", upstream_task_ids=["generate_data"])
# workflow.set_dependencies("visualize_data", upstream_task_ids=["process_data"])
# workflow.set_dependencies("send_report", upstream_task_ids=["visualize_data"])


def run_workflow():
    """Run the workflow tasks in sequence."""
    print("Running data processing workflow...")
    
    # Execute tasks in sequence
    result_generate = workflow.execute_task("generate_data")
    result_process = workflow.execute_task("process_data", data_path=result_generate["data_path"])
    result_visualize = workflow.execute_task(
        "visualize_data", 
        processed_path=result_process["processed_path"],
        stats_path=result_process["stats_path"]
    )
    result_report = workflow.execute_task(
        "send_report",
        scatter_plot=result_visualize["scatter_plot"],
        histogram=result_visualize["histogram"]
    )
    
    print("Workflow completed successfully!")
    return result_report


if __name__ == "__main__":
    # Run the workflow
    result = run_workflow()
    print(f"Final result: {result}")
    
    # If you want to run this as an Airflow DAG, you would export the DAG:
    # dag = workflow.get_dag()
    # This dag object can then be placed in your Airflow DAGs folder 