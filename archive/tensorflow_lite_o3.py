import os
import numpy as np
import tensorflow as tf
import pandas as pd
import random
from sklearn.model_selection import KFold

# ------------------------
# 1. Utility Functions
# ------------------------

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)

def load_data_from_file(file_path):
    """Load a single file (trial) using pandas and return as a numpy array."""
    # Assumes the file is space-separated with no header
    data = pd.read_csv(file_path, sep=" ", header=None)
    return data.values

def pad_or_truncate(seq, target_length=40):
    """
    Pad the sequence with zeros (or truncate) to ensure a fixed length.
    Assumes seq is a 2D numpy array of shape (T, features).
    Only the first 'target_length' rows are kept.
    """
    T = seq.shape[0]
    if T >= target_length:
        return seq[:target_length]
    else:
        pad_width = target_length - T
        pad_array = np.zeros((pad_width, seq.shape[1]))
        return np.vstack([seq, pad_array])

def load_gesture_data(gesture_folder, target_length=40):
    """
    Load all trial files from one gesture folder, padding/truncating each to target_length.
    """
    files = sorted([os.path.join(gesture_folder, f)
                    for f in os.listdir(gesture_folder) if f.endswith('.txt')])
    trials = []
    for file in files:
        seq = load_data_from_file(file)
        seq = pad_or_truncate(seq, target_length)
        trials.append(seq)
    return trials

def load_all_gestures(base_path, target_length=40):
    """
    Load data from all gesture folders.
    Assumes each gesture folder is named with an underscore and label at the end (e.g. "gesture_3").
    Returns:
        all_data: numpy array of shape (num_samples, target_length, 130)
        all_labels: numpy array of labels (as integers)
    """
    gesture_folders = [os.path.join(base_path, d)
                       for d in os.listdir(base_path)
                       if os.path.isdir(os.path.join(base_path, d))]
    all_data = []
    all_labels = []
    for folder in gesture_folders:
        trials = load_gesture_data(folder, target_length)
        # Extract label from folder name (assumes label is after the last underscore)
        gesture_label = int(folder.split('_')[-1])
        for trial in trials:
            all_data.append(trial)
            all_labels.append(gesture_label)
    return np.array(all_data), np.array(all_labels)

def load_predict_data(predict_path, target_length=40):
    """
    Load files for prediction (no labels).
    Returns:
        all_data: numpy array of shape (num_samples, target_length, 130)
        filenames: list of file names
    """
    files = sorted([os.path.join(predict_path, f) for f in os.listdir(predict_path) if f.endswith('.txt')])
    all_data = []
    filenames = []
    for file in files:
        seq = load_data_from_file(file)
        seq = pad_or_truncate(seq, target_length)
        all_data.append(seq)
        filenames.append(os.path.basename(file))
    return np.array(all_data), filenames

# ------------------------
# 2. Build the Model
# ------------------------

def build_model(input_shape, hidden_dim, output_dim, dropout_rate=0.2, weight_decay=0.0):
    """
    Build a CNN model that accepts a 40x130 input (with 1 channel).
    The architecture uses only TFLite built-in operators.
    """
    inputs = tf.keras.Input(shape=input_shape)  # Expected shape: (40, 130, 1)

    # First convolution block
    x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu',
                               kernel_regularizer=(tf.keras.regularizers.l2(weight_decay)
                                                   if weight_decay else None))(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Second convolution block
    x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu',
                               kernel_regularizer=(tf.keras.regularizers.l2(weight_decay)
                                                   if weight_decay else None))(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hidden_dim, activation='relu',
                              kernel_regularizer=(tf.keras.regularizers.l2(weight_decay)
                                                  if weight_decay else None))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(output_dim, activation='softmax',
                                    kernel_regularizer=(tf.keras.regularizers.l2(weight_decay)
                                                        if weight_decay else None))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# ------------------------
# 3. Hyperparameter Input Functions
# ------------------------

def get_hyperparameters():
    """
    Interactively ask for hyperparameter values.
    (num_layers has been removed since the model is a simple CNN.)
    """
    try:
        k_fold = int(input("Enter k_fold (default: 5; enter 0 to disable k-fold): "))
    except:
        k_fold = 5

    try:
        hidden_dim = int(input("Enter hidden_dim (options: 64,128,256, default: 64): "))
        if hidden_dim <= 0:
            print("hidden_dim must be > 0. Using default value 64.")
            hidden_dim = 64
    except:
        hidden_dim = 64

    try:
        epochs = int(input("Enter epochs (options: 50,100,150, default: 50): "))
        if epochs <= 0:
            print("epochs must be > 0. Using default value 50.")
            epochs = 50
    except:
        epochs = 50

    try:
        batch_size = int(input("Enter batch_size (options: 16,32,64, default: 32): "))
        if batch_size <= 0:
            print("batch_size must be > 0. Using default value 32.")
            batch_size = 32
    except:
        batch_size = 32

    try:
        dropout_rate = float(input("Enter dropout_rate (options: 0.2,0.3,0.5, default: 0.2): "))
        if dropout_rate <= 0:
            print("dropout_rate must be > 0. Using default value 0.2.")
            dropout_rate = 0.2
    except:
        dropout_rate = 0.2

    try:
        weight_decay = float(input("Enter weight_decay (options: 0,0.0001,0.001, default: 0): "))
    except:
        weight_decay = 0.0

    try:
        learning_rate = float(input(
            "Enter learning_rate (options: 0.001,0.0005,0.0001, default: 0.001; enter 0 to disable setting a custom learning rate): "))
    except:
        learning_rate = 0.001
    if learning_rate == 0:
        learning_rate = None

    try:
        step_size = int(input("Enter step_size (options: 10,20,30, default: 10; enter 0 to disable scheduler): "))
    except:
        step_size = 10
    if step_size == 0:
        step_size = None

    try:
        gamma = float(input("Enter gamma (options: 0.1,0.5, default: 0.1; enter 0 to disable scheduler): "))
    except:
        gamma = 0.1
    if gamma == 0:
        gamma = None

    return {
        "k_fold": k_fold,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "dropout_rate": dropout_rate,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "step_size": step_size,
        "gamma": gamma
    }

# ------------------------
# 4. Main Script
# ------------------------

def main():
    set_seed(66)
    select = input('Enter option number (Enter 1 to train the model, or t to test the model): ').strip()

    # Define paths (adjust these to your environment)
    base_path = 'C:/dev/phd/ai'
    train_path = 'C:/dev/phd/ai/training_data'
    test_path = 'C:/dev/phd/ai/testing_data'
    predict_path = 'C:/dev/phd/predict/'
    model_save_path = 'C:/dev/phd/models/'
    tflite_save_path = 'C:/dev/phd/tflite/'

    # Settings for fixed-size input.
    # Each file is expected to be a 40-row x 130-column matrix.
    rows = 40         # number of rows (or “time steps”)
    cols = 130        # number of columns (features)
    channels = 1
    input_shape = (rows, cols, channels)  # (40, 130, 1)

    if select == '1':
        # ------------------------
        # Training with optional hyperparameter selection
        # ------------------------
        use_default = input(
            "Do you want to use the default hyperparameters? (Enter any value other than 0 for YES, or 0 for NO): ").strip()
        if use_default != '0':
            hyperparams = {
                "k_fold": 0,
                "hidden_dim": 32,
                "epochs": 50,
                "batch_size": 32,
                "dropout_rate": 0.0,
                "weight_decay": 0.0,
                "learning_rate": None,
                "step_size": None,
                "gamma": None
            }
        else:
            hyperparams = get_hyperparameters()
        print("\nUsing hyperparameters:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")

        print("\nLoading training and test data...")
        train_data, train_labels = load_all_gestures(train_path, rows)
        test_data, test_labels = load_all_gestures(test_path, rows)

        # The CSV files already have 130 columns. Just add a channel dimension.
        train_data = np.expand_dims(train_data, -1)  # shape becomes (num_samples, 40, 130, 1)
        test_data = np.expand_dims(test_data, -1)

        # Determine number of classes (output dimension)
        output_dim = len(np.unique(train_labels))
        print(f"Number of classes: {output_dim}")

        # Initialize scheduler_callback to None so it's defined regardless of k_fold usage.
        scheduler_callback = None

        # ------------------------
        # (Optional) k-fold cross-validation
        # ------------------------
        if hyperparams["k_fold"]:
            print("\nStarting k-fold cross-validation...")
            kf = KFold(n_splits=hyperparams["k_fold"], shuffle=True, random_state=54)
            fold_no = 1
            val_accuracies = []

            if hyperparams["learning_rate"] is not None and hyperparams["step_size"] is not None and hyperparams["gamma"] is not None:
                def scheduler(epoch, lr):
                    if epoch > 0 and epoch % hyperparams["step_size"] == 0:
                        return lr * hyperparams["gamma"]
                    return lr
                scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
            else:
                scheduler_callback = None

            for train_index, val_index in kf.split(train_data):
                print(f"\n--- Training fold {fold_no} ---")
                X_train_fold = train_data[train_index]
                y_train_fold = train_labels[train_index]
                X_val_fold = train_data[val_index]
                y_val_fold = train_labels[val_index]

                model = build_model(input_shape, hyperparams["hidden_dim"], output_dim,
                                    dropout_rate=hyperparams["dropout_rate"],
                                    weight_decay=hyperparams["weight_decay"])

                if hyperparams["learning_rate"] is not None:
                    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams["learning_rate"])
                else:
                    optimizer = tf.keras.optimizers.Adam()

                model.compile(optimizer=optimizer,
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                callbacks_list = [scheduler_callback] if scheduler_callback is not None else []
                model.fit(X_train_fold, y_train_fold,
                          epochs=hyperparams["epochs"],
                          batch_size=hyperparams["batch_size"],
                          callbacks=callbacks_list,
                          verbose=1)

                loss, acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
                print(f"Fold {fold_no} Validation Accuracy: {acc * 100:.2f}%")
                val_accuracies.append(acc)
                fold_no += 1

            avg_val_accuracy = np.mean(val_accuracies)
            print(f"\nAverage Validation Accuracy across folds: {avg_val_accuracy * 100:.2f}%")
        else:
            print("\nk-fold cross-validation disabled; training directly on full training data.")

        # ------------------------
        # Train Final Model on Full Training Data
        # ------------------------
        print("\nTraining final model on full training data...")
        final_model = build_model(input_shape, hyperparams["hidden_dim"], output_dim,
                                  dropout_rate=hyperparams["dropout_rate"],
                                  weight_decay=hyperparams["weight_decay"])
        if hyperparams["learning_rate"] is not None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams["learning_rate"])
        else:
            optimizer = tf.keras.optimizers.Adam()
        final_model.compile(optimizer=optimizer,
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
        callbacks_list = [scheduler_callback] if scheduler_callback is not None else []

        final_model.fit(train_data, train_labels,
                        epochs=hyperparams["epochs"],
                        batch_size=hyperparams["batch_size"],
                        callbacks=callbacks_list,
                        verbose=1)
        test_loss, test_accuracy = final_model.evaluate(test_data, test_labels)
        print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

        # ------------------------
        # Save the Keras model.
        # ------------------------
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        model_filename = os.path.join(model_save_path, 'best_model.h5')
        final_model.save(model_filename)
        print(f"Keras model saved to {model_filename}")

        # ------------------------
        # Convert to TensorFlow Lite model.
        # ------------------------
        converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
        converter._experimental_lower_tensor_list_ops = False
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()

        if not os.path.exists(tflite_save_path):
            os.makedirs(tflite_save_path)
        tflite_model_filename = os.path.join(tflite_save_path, 'best_model.tflite')
        with open(tflite_model_filename, 'wb') as f:
            f.write(tflite_model)
        print(f"TensorFlow Lite model saved to {tflite_model_filename}")

    elif select.lower() == 't':
        # ------------------------
        # Inference on new data using the saved Keras model.
        # ------------------------
        model_filename = os.path.join(model_save_path, 'best_model.h5')
        if not os.path.exists(model_filename):
            print("Model file not found.")
            return

        model = tf.keras.models.load_model(model_filename)
        model.summary()

        predict_data, filenames = load_predict_data(predict_path, rows)
        predict_data = np.expand_dims(predict_data, -1)  # shape (num_samples, 40, 130, 1)

        predictions = model.predict(predict_data)
        predicted_classes = np.argmax(predictions, axis=1)

        for fname, pred in zip(filenames, predicted_classes):
            print(f'File {fname}: Predicted Gesture: {pred}')
    else:
        print("Invalid option.")

if __name__ == '__main__':
    main()
