import argparse
import joblib

from src.DF_itunes_amazon.data_preprocessor import DataPreprocessor
from src.DF_penumbra.training_magellan import MagellanTrainer


if __name__ == "__main__":
    model_choices = ["dt", "svm", "rf", "lg", "ln", "nb", "xgb"]

    parser = argparse.ArgumentParser(
        description="Run different functions based on input parameters."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="task name:{m_training}",
        metavar="",
    )
    parser.add_argument(
        "--m_model",
        choices=model_choices,
        type=str,
        default=None,
        help="Magellan model name",
        metavar="",
    )

    args = parser.parse_args()

    if args.task == "m_training":
        """
        Prepare the training data, train the Magellan model, and evaluate the model.
        Note: copy data from https://github.com/anhaidgroup/deepmatcher/tree/master/examples/sample_data/itunes-amazon
            to "itunes-amazon" folder in this project
        """
        if not args.m_model:
            raise ValueError("Please specify the model to use for training.")

        data_dir = "itunes-amazon"
        dp = DataPreprocessor(data_dir=data_dir)
        ltable, rtable, data = dp.prepare_training_data()

        mt = MagellanTrainer(ltable, rtable, data, model=args.m_model, training=True)
        mt.train_model()

        print("Evaluating the model...")
        preds = mt.predict()
        mt.evaluate(preds)

        print("Displaying feature importance...")
        mt.retrieve_feature_importance()

    elif args.task == "predict-all":
        """
        Prerequisits:
        - model.pkl file should be available from the training.

        Run predictions on the entire data except for the training data
        """
        if not args.m_model:
            raise ValueError("Please specify the model used for training.")

        print(f"Running {args.m_model} over the entire data...")

        data_dir = "itunes-amazon"
        model = joblib.load("model.pkl")
        dp = DataPreprocessor(data_dir)

        df = dp.prepare_all_data()
        print("df", df)

        A, B, C = dp._load_data(df)

        mt = MagellanTrainer(A, B, C, model)
        preds = mt.predict(C, all=True)
