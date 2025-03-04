"""
THIS FILE IS DEPRECATED AND PERFECT WILL BE USED IN THE FUTURE.
THIS FILE IS LEFT JUST IN CASE OF SPECIAL NEEDS.

PyOmnix Workflow Framework based on Apache Airflow

This module provides a flexible workflow framework using Apache Airflow.
It allows for easy creation and management of workflows in other modules or Python files.
"""

import datetime
from typing import Dict, List, Callable, Union, Any
import functools

# Airflow imports
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import TaskInstance
from airflow.utils.context import Context

# Import PyOmnix logger if available
from .omnix_logger import setup_logger
logger = setup_logger(__name__)

class WorkflowManager:
    """
    A manager class for creating and managing Airflow workflows.
    
    This class provides a simplified interface for creating Airflow DAGs and tasks,
    making it easy to integrate workflow capabilities into any Python module.
    """
    
    def __init__(self, 
                 dag_id: str, 
                 description: str = None,
                 start_date: datetime.datetime = days_ago(1),
                 schedule_interval: Union[str, datetime.timedelta] = None,
                 catchup: bool = False,
                 tags: List[str] = None,
                 default_args: Dict = None):
        """
        Initialize a new WorkflowManager.
        
        Args:
            dag_id: Unique identifier for the DAG
            description: Optional description of the DAG
            start_date: Date from which the DAG should start running
            schedule_interval: How often the DAG should run
            catchup: Whether to backfill missed DAG runs
            tags: List of tags for the DAG
            default_args: Default arguments for all tasks in the DAG
        """
        if default_args is None:
            default_args = {
                'owner': 'PyOmnix',
                'depends_on_past': False,
                'email_on_failure': False,
                'email_on_retry': False,
                'retries': 1,
                'retry_delay': datetime.timedelta(minutes=5),
            }
        
        if tags is None:
            tags = ['PyOmnix']
        elif 'PyOmnix' not in tags:
            tags.append('PyOmnix')
            
        self.dag = DAG(
            dag_id=dag_id,
            description=description,
            default_args=default_args,
            start_date=start_date,
            schedule_interval=schedule_interval,
            catchup=catchup,
            tags=tags,
        )
        
        self.tasks = {}
        self.task_dependencies = {}
        
    def add_task(self,
                 task_id: str,
                 python_callable: Callable,
                 call_kwargs: Dict = None,
                 trigger_rule: str = 'all_success',
                 retries: int = None,
                 retry_delay: datetime.timedelta = None,
                 depends_on_past: bool = None,
                 **kwargs) -> PythonOperator:
        """
        Add a task to the workflow.
        
        Args:
            task_id: Unique identifier for the task
            python_callable: The Python function to execute
            call_kwargs: Keyword arguments to pass to the function
            trigger_rule: Rule defining when the task should be triggered
            retries: Number of retries for this specific task
            retry_delay: Delay between retries for this specific task
            depends_on_past: Whether this task depends on past runs
            **kwargs: Additional keyword arguments for the PythonOperator
            
        Returns:
            The created PythonOperator task
        """
        if call_kwargs is None:
            call_kwargs = {}
        # Create a wrapper function that logs task execution
        @functools.wraps(python_callable)
        def task_wrapper(*args, **kwargs):
            ti = kwargs.get('ti')
            task_id = ti.task_id if ti else 'unknown'
            dag_id = ti.dag_id if ti else 'unknown'
            logger.info("Starting task '%s' in DAG '%s'", task_id, dag_id)
            try:
                result = python_callable(*args, **kwargs)
                logger.info("Task '%s' in DAG '%s' completed successfully", task_id, dag_id)
                return result
            except Exception as e:
                logger.error("Task '%s' in DAG '%s' failed: %s", task_id, dag_id, str(e))
                raise
        
        # Create task-specific arguments
        task_args = {}
        if retries is not None:
            task_args['retries'] = retries
        if retry_delay is not None:
            task_args['retry_delay'] = retry_delay
        if depends_on_past is not None:
            task_args['depends_on_past'] = depends_on_past
            
        # Create the task
        task = PythonOperator(
            task_id=task_id,
            python_callable=task_wrapper,
            op_kwargs=call_kwargs,
            trigger_rule=trigger_rule,
            dag=self.dag,
            **task_args,
            **kwargs
        )
        
        self.tasks[task_id] = task
        return task
    
    def set_dependencies(self, task_id: str, upstream_task_ids: List[str] = None, downstream_task_ids: List[str] = None):
        """
        Set dependencies between tasks.
        
        Args:
            task_id: The task for which to set dependencies
            upstream_task_ids: List of task IDs that should run before this task
            downstream_task_ids: List of task IDs that should run after this task
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found in workflow")
            
        task = self.tasks[task_id]
        
        if upstream_task_ids:
            for upstream_id in upstream_task_ids:
                if upstream_id not in self.tasks:
                    raise ValueError(f"Upstream task '{upstream_id}' not found in workflow")
                self.tasks[upstream_id] >> task
                
        if downstream_task_ids:
            for downstream_id in downstream_task_ids:
                if downstream_id not in self.tasks:
                    raise ValueError(f"Downstream task '{downstream_id}' not found in workflow")
                task >> self.tasks[downstream_id]
    
    def get_dag(self) -> DAG:
        """
        Get the Airflow DAG object.
        
        Returns:
            The Airflow DAG object
        """
        return self.dag
    
    def get_task(self, task_id: str) -> PythonOperator:
        """
        Get a task by its ID.
        
        Args:
            task_id: The ID of the task to get
            
        Returns:
            The task with the given ID
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found in workflow")
        return self.tasks[task_id]
    
    def execute_task(self, task_id: str, execution_date: datetime.datetime = None, **kwargs) -> Any:
        """
        Execute a single task directly (without running the entire DAG).
        
        Args:
            task_id: The ID of the task to execute
            execution_date: The execution date to use
            **kwargs: Additional keyword arguments to pass to the task
            
        Returns:
            The result of the task execution
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found in workflow")
            
        if execution_date is None:
            execution_date = datetime.datetime.now()
            
        task = self.tasks[task_id]
        ti = TaskInstance(task=task, execution_date=execution_date)
        
        # Merge kwargs with op_kwargs
        op_kwargs = task.op_kwargs.copy() if hasattr(task, 'op_kwargs') else {}
        op_kwargs.update(kwargs)
        op_kwargs['ti'] = ti
        
        # Execute the task
        context = Context(ti=ti, execution_date=execution_date)
        return task.execute(context=context)


class TaskDecorator:
    """
    A decorator class for creating workflow tasks from functions.
    
    This class provides decorators that can be used to convert regular Python
    functions into workflow tasks.
    """
    
    @staticmethod
    def task(workflow_manager: WorkflowManager,
             task_id: str = None,
             op_kwargs: Dict = None,
             **task_kwargs) -> Callable:
        """
        Decorator to convert a function into a workflow task.
        
        Args:
            workflow_manager: The WorkflowManager to add the task to
            task_id: Optional task ID (defaults to function name)
            op_kwargs: Keyword arguments to pass to the function
            **task_kwargs: Additional keyword arguments for the task
            
        Returns:
            A decorator function
        """
        def decorator(func: Callable) -> Callable:
            nonlocal task_id
            if task_id is None:
                task_id = func.__name__
                
            # Add the task to the workflow
            workflow_manager.add_task(
                task_id=task_id,
                python_callable=func,
                call_kwargs=op_kwargs,
                **task_kwargs
            )
            
            # Return the original function unchanged
            return func
        
        return decorator


# Convenience function to create a workflow manager
def create_workflow(dag_id: str, **kwargs) -> WorkflowManager:
    """
    Create a new workflow manager.
    
    Args:
        dag_id: Unique identifier for the DAG
        **kwargs: Additional keyword arguments for the WorkflowManager
        
    Returns:
        A new WorkflowManager instance
    """
    return WorkflowManager(dag_id=dag_id, **kwargs)


# Example usage
if __name__ == "__main__":
    # Create a workflow
    workflow = create_workflow(
        dag_id="example_workflow",
        description="An example workflow",
        schedule_interval=datetime.timedelta(days=1)
    )
    
    # Define tasks
    def task1(ti, **kwargs):
        print("Executing task 1")
        return {"task1_result": "success"}
    
    def task2(ti, task1_result, **kwargs):
        print(f"Executing task 2 with input: {task1_result}")
        return {"task2_result": "processed " + task1_result}
    
    # Add tasks to the workflow
    t1 = workflow.add_task(
        task_id="task1",
        python_callable=task1
    )
    
    t2 = workflow.add_task(
        task_id="task2",
        python_callable=task2,
        call_kwargs={"task1_result": "{{ ti.xcom_pull(task_ids='task1')['task1_result'] }}"}
    )
    
    # Set dependencies
    t1 >> t2
    
    # Alternative way to set dependencies
    # workflow.set_dependencies("task2", upstream_task_ids=["task1"])
    
    # Get the DAG
    dag = workflow.get_dag()
