# PyOmnix Workflow Framework

The PyOmnix Workflow Framework provides a flexible and easy-to-use interface for creating and managing workflows using Apache Airflow. This framework allows you to define, organize, and execute complex workflows in your Python applications.

## Features

- Simple API for creating Airflow DAGs and tasks
- Task dependency management
- Task execution tracking and logging
- Support for both scheduled and on-demand workflow execution
- Decorator-based task definition
- XCom data passing between tasks

## Installation

To use the workflow framework, you need to install PyOmnix with the workflow extras:

```bash
pip install PyOmnix[workflow]
```

## Basic Usage

Here's a simple example of how to use the workflow framework:

```python
import datetime
from PyOmnix.flow import create_workflow

# Create a workflow
workflow = create_workflow(
    dag_id="my_workflow",
    description="A simple workflow example",
    schedule_interval=datetime.timedelta(days=1)
)

# Define task functions
def task1(ti, **kwargs):
    print("Executing task 1")
    return {"result": "Task 1 completed"}

def task2(ti, result, **kwargs):
    print(f"Executing task 2 with input: {result}")
    return {"final_result": f"Processed: {result}"}

# Add tasks to the workflow
t1 = workflow.add_task(
    task_id="task1",
    python_callable=task1
)

t2 = workflow.add_task(
    task_id="task2",
    python_callable=task2,
    op_kwargs={"result": "{{ ti.xcom_pull(task_ids='task1')['result'] }}"}
)

# Set dependencies
t1 >> t2

# Execute the workflow
result_task1 = workflow.execute_task("task1")
result_task2 = workflow.execute_task("task2", result=result_task1["result"])
```

## Using Task Decorators

You can also use decorators to define tasks:

```python
from PyOmnix.flow import create_workflow, TaskDecorator

# Create a workflow
workflow = create_workflow(dag_id="decorator_workflow")

# Define tasks using decorators
@TaskDecorator.task(workflow_manager=workflow, task_id="first_task")
def first_task(ti, **kwargs):
    return {"data": "some data"}

@TaskDecorator.task(
    workflow_manager=workflow, 
    task_id="second_task",
    op_kwargs={"data": "{{ ti.xcom_pull(task_ids='first_task')['data'] }}"}
)
def second_task(ti, data, **kwargs):
    return {"processed_data": f"Processed: {data}"}

# Set dependencies
workflow.set_dependencies("second_task", upstream_task_ids=["first_task"])
```

## Integration with Airflow

To use your workflow with Airflow, you can export the DAG:

```python
# Create and configure your workflow
workflow = create_workflow(dag_id="airflow_workflow")
# ... add tasks and dependencies ...

# Export the DAG for Airflow
dag = workflow.get_dag()

# Save this file in your Airflow DAGs folder
```

## Advanced Features

### Task Retry Configuration

```python
workflow.add_task(
    task_id="retry_task",
    python_callable=my_function,
    retries=3,
    retry_delay=datetime.timedelta(minutes=5)
)
```

### Custom Trigger Rules

```python
workflow.add_task(
    task_id="trigger_task",
    python_callable=my_function,
    trigger_rule="one_success"  # Run if at least one upstream task succeeds
)
```

### Task Dependencies

```python
# Method 1: Using operators
task1 >> task2 >> task3

# Method 2: Using the set_dependencies method
workflow.set_dependencies(
    task_id="task2",
    upstream_task_ids=["task1"],
    downstream_task_ids=["task3"]
)
```

## Complete Example

For a complete example of how to use the workflow framework, see the [workflow_example.py](../examples/workflow_example.py) file in the examples directory.

## API Reference

### WorkflowManager

The main class for creating and managing workflows.

- `__init__(dag_id, description=None, start_date=days_ago(1), schedule_interval=None, catchup=False, tags=None, default_args=None)`: Initialize a new workflow manager
- `add_task(task_id, python_callable, op_kwargs=None, trigger_rule='all_success', **kwargs)`: Add a task to the workflow
- `set_dependencies(task_id, upstream_task_ids=None, downstream_task_ids=None)`: Set dependencies between tasks
- `get_dag()`: Get the Airflow DAG object
- `get_task(task_id)`: Get a task by its ID
- `execute_task(task_id, execution_date=None, **kwargs)`: Execute a single task directly

### TaskDecorator

A class providing decorators for creating workflow tasks from functions.

- `task(workflow_manager, task_id=None, op_kwargs=None, **task_kwargs)`: Decorator to convert a function into a workflow task

### Helper Functions

- `create_workflow(dag_id, **kwargs)`: Convenience function to create a workflow manager 