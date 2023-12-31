import pytorch_lightning as pl
import torch
import pandas as pd
from torch.utils.data import DataLoader

from classification_task import *
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
import numpy as np
if __name__ == "__main__":
    
    device = "gpu" if torch.cuda.is_available() else "cpu"

    test_csv = pd.read_csv(
        "vindr-spinexr-dataset/test_classify.csv", index_col=0)
    test_path = "vindr-spinexr-dataset/test_images_png_224/"

    test_dataset = DataLoader(ClsLoader(test_path, test_csv, typeData="test"), batch_size=128,
                              num_workers=4, prefetch_factor=64)
    model = ClassifyNet_vs2()
    classifier = Classifier(model=model, class_weight=config['CLASS_WEIGHT'],
                            num_classes=config["NUM_CLASS"], learning_rate=config["LEARNING_RATE"])
    PARAMS = {"accelerator": device, "devices": 1, "benchmark": True, "enable_progress_bar": True,
              #   "callbacks" : [progress_bar],
              #    "overfit_batches" :1,
              "logger": False,
              #   "callbacks": [check_point, logger, lr_monitor, swa],
              "log_every_n_steps": 1, "num_sanity_val_steps": 2, "max_epochs": 15,
              #   "precision":16,
              }

    trainer = pl.Trainer(**PARAMS)
    chk_path = "classification_task/weights/best_weight_cls.ckpt"
    classifier = classifier.load_from_checkpoint(checkpoint_path=chk_path, model=ClassifyNet_vs2(), class_weight=config['CLASS_WEIGHT'],
                                                 num_classes=config["NUM_CLASS"], learning_rate=config["LEARNING_RATE"])

    # trainer.test(classifier, test_dataset)
    # predictions = trainer.predict(classifier, dataloaders=test_dataset)
    # print(predictions.shape)

    ##################### evaluate #############################################
    labels = np.array(test_csv["abnormal"])
    ids = test_csv["image_id"]
    pred = np.vstack(trainer.predict(classifier, test_dataset))
    y_pred = np.argmax(pred, axis=1)
    # print(pred.shape)
    print("f1:", f1_score(labels, y_pred))
    tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    print("specificity:", specificity)
    print("sensitivity:", sensitivity)
    print("roc_auc_score:", roc_auc_score(labels, pred[:,1]))
    print("done")