# src/main.py
# Use relative import within the 'src' package
from . import pipeline

def main():
    """
    Main entry point for the energy prediction pipeline.
    """
    # Run the pipeline with default settings (or potentially load settings from config/args)
    pipeline.run_pipeline(
        # You can override defaults here if needed, e.g.:
        # n_runs=5,
        # data_sample_size=20000
    )

if __name__ == "__main__":
    # Ensure script runs only when executed directly OR via python -m src.main
    main()
