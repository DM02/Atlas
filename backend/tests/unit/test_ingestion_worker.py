def test_worker_settings_registers_run_ingestion() -> None:
    from app.workers.ingestion_worker import WorkerSettings, run_ingestion

    assert run_ingestion in WorkerSettings.functions
