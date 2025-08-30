from src.logger import get_logger
from src.data_processing import DataProcessing
from src.model_training import ModelTraining

logger = get_logger(__name__)

if __name__ == "__main__":

        logger.info("Starting data processing Pipeline...")
        input_path = "artifacts/raw/data.csv"
        output_path = "artifacts/processed"

        data_processor = DataProcessing(input_path, output_path)
        data_processor.run()


        processed_data_path = "artifacts/processed"
        model_output_path = "artifacts/models"

        model_trainer = ModelTraining(processed_data_path, model_output_path)
        model_trainer.run()

        logger.info("Model training pipeline completed.")
