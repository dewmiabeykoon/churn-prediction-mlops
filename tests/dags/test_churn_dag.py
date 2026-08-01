"""Tests for the customer churn Airflow DAG."""

import os
from contextlib import contextmanager

import pytest
from airflow.models import DagBag


@contextmanager
def suppress_logging(namespace):
    import logging

    logger = logging.getLogger(namespace)
    old_value = logger.disabled
    logger.disabled = True
    try:
        yield
    finally:
        logger.disabled = old_value


def get_import_errors():
    with suppress_logging("airflow"):
        dag_bag = DagBag(include_examples=False)

        def strip_path_prefix(path):
            return os.path.relpath(path, os.environ.get("AIRFLOW_HOME", os.getcwd()))

        return [(None, None)] + [
            (strip_path_prefix(path), error.strip())
            for path, error in dag_bag.import_errors.items()
        ]


def get_dags():
    with suppress_logging("airflow"):
        dag_bag = DagBag(include_examples=False)

    def strip_path_prefix(path):
        return os.path.relpath(path, os.environ.get("AIRFLOW_HOME", os.getcwd()))

    return [(dag_id, dag, strip_path_prefix(dag.fileloc)) for dag_id, dag in dag_bag.dags.items()]


@pytest.mark.parametrize("rel_path,rv", get_import_errors(), ids=[x[0] for x in get_import_errors()])
def test_file_imports(rel_path, rv):
    if rel_path and rv:
        raise AssertionError(f"{rel_path} failed to import with message \n {rv}")


@pytest.mark.parametrize("dag_id,dag,fileloc", get_dags(), ids=[x[2] for x in get_dags()])
def test_dag_has_tags(dag_id, dag, fileloc):
    assert dag.tags, f"{dag_id} in {fileloc} has no tags"


@pytest.mark.parametrize("dag_id,dag,fileloc", get_dags(), ids=[x[2] for x in get_dags()])
def test_churn_dag_task_count(dag_id, dag, fileloc):
    if dag_id != "customer_churn_dag":
        return
    assert len(dag.tasks) == 6, f"{dag_id} should contain 6 tasks"
