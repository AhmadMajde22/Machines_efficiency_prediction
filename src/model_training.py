import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from src.logger import get_logger
from src.custome_exception import CustomException

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self,processed_data_path,model_output_path):
        self.processed_path = processed_data_path
        self.model_path = model_output_path
        self.clf = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        os.makedirs(self.model_path,exist_ok=True)

        logger.info("Model training initialized.")


    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_path,"X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_path,"X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_path,"y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_path,"y_test.pkl"))
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Error loading data", e)

    def train_model(self):
        try:
            self.clf = LogisticRegression(random_state=42,max_iter=1000)
            self.clf.fit(self.X_train,self.y_train)

            joblib.dump(self.clf,os.path.join(self.model_path,"model.pkl"))

            logger.info("Model training and saving completed successfully.")
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise CustomException("Error training model", e)


    def evaluate_model(self):
        try:
            y_pred = self.clf.predict(self.X_test)

            accuracy = accuracy_score(self.y_test,y_pred)
            recall = recall_score(self.y_test,y_pred,average='weighted')
            precision = precision_score(self.y_test,y_pred,average='weighted')
            f1 = f1_score(self.y_test,y_pred,average='weighted')

            logger.info(f"Model Evaluation Metrics:\nAccuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}\nF1-Score: {f1}")

            logger.info("Model evaluation completed successfully.")

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise CustomException("Error evaluating model", e)


    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()


if __name__ == "__main__":
    processed_data_path = "artifacts/processed"
    model_output_path = "artifacts/models"

    model_trainer = ModelTraining(processed_data_path, model_output_path)
    model_trainer.run()
