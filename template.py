import os

def create_files_and_directories(file_list):
    """
    Create the required directories and files based on the provided list.
    
    Parameters:
    file_list (list): List of file paths to create.
    """
    for filepath in file_list:
        filedir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        # Create the directory if it does not exist
        if filedir and not os.path.isdir(filedir):
            os.makedirs(filedir)
            print(f"Creating directory: {filedir} for the file {filename}")

        # Create the file if it does not exist or is empty
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            open(filepath, 'a').close()
            print(f"Creating empty file: {filepath}")
        else:
            print(f"{filename} already exists")


if __name__ == "__main__":

    # List of files to be created
    list_of_files = [
        "src/__init__.py",

        "src/data/__init__.py",
        "src/data/data_ingestion.py",

        "src/features/__init__.py",
        "src/features/data_transformation.py",

        "src/models/__init__.py",
        "src/models/model_trainer.py",

        "src/monitoring/__init__.py",
        "src/monitoring/model_monitoring.py",

        "src/utils/__init__.py",
        "src/utils/basic_util.py",

        "src/exceptions/__init__.py",
        "src/exceptions/exception.py",

        "src/logger/__init__.py",
        "src/logger/base_logger.py",

        "src/pipelines/__init__.py",
        "src/pipelines/training_pipeline.py",
        "src/pipelines/prediction_pipeline.py",
        
        "main.py"
    ]

    # Calling the function to create files and directories
    create_files_and_directories(list_of_files)