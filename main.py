from services.experiment_orchestrator import ExperimentOrchestrator

if __name__ == "__main__":
    ExperimentOrchestrator("config/templates/config.json").run()