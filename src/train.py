import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger

def train_neural_network(model, train_data, val_data, optimizer, loss, metrics, epochs, verbose, save_path, model_name='model.h5', batch_size=32,
                        early_stopping_monitor='val_precision', early_stopping_mode='max', early_stopping_patience=30,
                        reduce_lr_monitor='val_precision', reduce_lr_mode='max', reduce_lr_patience=10, reducelr_factor=0.1):

    early_stopping = EarlyStopping(monitor=early_stopping_monitor, mode=early_stopping_mode, patience=early_stopping_patience, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor=reduce_lr_monitor, mode=reduce_lr_mode, factor=reducelr_factor, patience=reduce_lr_patience, verbose=1)
    csv_logger = CSVLogger(os.path.join(save_path, 'training.csv'))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    hist = model.fit(x=train_data, validation_data=val_data, epochs=epochs, verbose=verbose, callbacks=[early_stopping, reduce_lr, csv_logger], batch_size=batch_size)
     
    predicted_values = model.predict(val_data)
    predicted_values = np.argmax(predicted_values, axis=-1)

    classif_mat = confusion_matrix(val_data.classes, predicted_values)
    classif_mat_df = pd.DataFrame(classif_mat, index=['NORMAL', 'PNEUMONIA'], columns=['NORMAL', 'PNEUMONIA'])

    # kreiranje i snimanje matrice konfuzije

    plt.clf()
    sns.heatmap(classif_mat_df, annot=True, fmt='d', cbar=False)
    confusion_matrix_heatmap_save_path = os.path.join(save_path, 'confusion_matrix.png')
    
    save_confusion_matrix_path = os.path.join(save_path, 'confusion_matrix.png') 
    plt.savefig(save_confusion_matrix_path)

    # kreiranje i snimanje grafika u kome se porede gubici (loss) trening opservacija naspram validacionih opservacija

    plt.clf()

    plt.plot(hist.history['loss'], label='train loss')
    plt.plot(hist.history['val_loss'], label='validation loss')
    plt.legend(loc='upper right')
    plt.title('Loss vs validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    save_loss_plot = os.path.join(save_path, 'train_loss_vs_val_loss.png')
    plt.savefig(save_loss_plot)

    # kreiranje i snimanje grafika poredjenja tacnosti trening opservacija naspram validacionih opservacija (accuracy)

    plt.clf()
    plt.plot(hist.history['categorical_accuracy'], label='train accuracy')
    plt.plot(hist.history['val_categorical_accuracy'], label='validation accuracy')
    plt.legend(loc='lower right')
    plt.title('Train accuracy vs validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    save_accuracy_plot = os.path.join(save_path, 'train_accuracy_vs_val_accuracy.png')
    plt.savefig(save_accuracy_plot)

    # snimanje grafika poredjenja metrike preciznosti trening i validacionih opservacija (precision)

    plt.clf()
    plt.plot(hist.history['precision'], label='train precision')
    plt.plot(hist.history['val_precision'], label='validation precision')
    plt.legend(loc='lower right')
    plt.title('Train precision vs validation precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')

    save_precision_plot = os.path.join(save_path, 'train_precision_vs_val_precision.png')
    plt.savefig(save_precision_plot)

    # snimanje izvestaja (classification report) i tacnosti (accuracy)

    class_report = classification_report(val_data.classes, predicted_values)
    accuracy = accuracy_score(val_data.classes, predicted_values)

    classification_report_and_accuracy_path = os.path.join(save_path, 'classification_report_and_accuracy_path.txt')

    with open(classification_report_and_accuracy_path, 'w') as f:
        f.write(class_report)
        f.write('\n \n \n')
        f.write(f'accuracy: {accuracy}')

    # snimanje modela

    save_model_path = os.path.join(save_path, model_name)
    model.save(save_model_path) # include_optimizer=False

    # snimanje summary-a

    stdout = sys.stdout
    summary_path = os.path.join(save_path, 'model_summary.txt')
    with open(summary_path, 'w') as f:
        sys.stdout = f
        model.summary()
        sys.stdout = stdout