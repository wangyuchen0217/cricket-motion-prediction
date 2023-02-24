# Crickets_Walking_Motion_Prediction

model_predict.py
    def get_prediction_from_transformer
    return
    cannot reshape array of size 79631 into shape (3)

model_predict.py
    def get_results
        pred_test = y_scaler.inverse_transform(pred_test_scaled)
                Expected 2D array, got 1D array instead:
                array=[].
                Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
