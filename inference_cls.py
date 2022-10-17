import pytorch_lightning as pl
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from classification_task.dataset import ClsLoader
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
from classification_task.model import ClassifyNet_vs2, Classifier
from classification_task.utils_cls import load_config

import numpy as np
import pdb


config = load_config("classification_task/config.yaml")

path_dataset = "vindr-spinexr-dataset"

def find_threshold_4_det(test_csv, test_path=f"{path_dataset}/test_images_png_224/", config=config,
                         chk_path="classification_task/weights/ckpt0.8045.ckpt"):
    test_dataset = DataLoader(ClsLoader(test_path, test_csv, typeData="test"), batch_size=128,
                              num_workers=4, prefetch_factor=64)
    model = ClassifyNet_vs2()
    classifier = Classifier(model=model, class_weight=config['CLASS_WEIGHT'],
                            num_classes=config["NUM_CLASS"], learning_rate=config["LEARNING_RATE"])
    PARAMS = {"accelerator": 'gpu', "devices": 1, "benchmark": True, "enable_progress_bar": True,
              #   "callbacks" : [progress_bar],
              #    "overfit_batches" :1,
              "logger": False,
              #   "callbacks": [check_point, logger, lr_monitor, swa],
              "log_every_n_steps": 1, "num_sanity_val_steps": 2, "max_epochs": 15,
              #   "precision":16,
              }

    trainer = pl.Trainer(**PARAMS)
    classifier = classifier.load_from_checkpoint(checkpoint_path=chk_path, model=ClassifyNet_vs2(), class_weight=config['CLASS_WEIGHT'],
                                                 num_classes=config["NUM_CLASS"], learning_rate=config["LEARNING_RATE"])
    predictions = np.vstack(trainer.predict(
        classifier, dataloaders=test_dataset))[:, 1]  # get probs for abnormal

    fpr, tpr, thresholds = roc_curve(labels, predictions)
    J = tpr - fpr
    thresh_i = np.argmax(J)
    threshold = thresholds[thresh_i]
    return threshold

if __name__ == "__main__":
    device = "gpu" if torch.cuda.is_available() else "cpu"
    test_csv = pd.read_csv(
        f"{path_dataset}/test_classify.csv", index_col=0)
    test_path = f"{path_dataset}/test_images_png_224/"

    config = load_config("classification_task/config.yaml")

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
    chk_path = "classification_task/weights/best_weight.ckpt"
    classifier = classifier.load_from_checkpoint(checkpoint_path=chk_path, model=ClassifyNet_vs2(), class_weight=config['CLASS_WEIGHT'],
                                                 num_classes=config["NUM_CLASS"], learning_rate=config["LEARNING_RATE"])
    classifier.eval()
    # labels = np.array(test_csv["abnormal"])
    # ids = test_csv["image_id"]

    ################################################################################

    ### NORMAL IMAGES
    # path = f"{path_dataset}/test_pngs/1e06c2a8705f5dc0beb77cc55dca45bb.png"

    ### ABNORMAL IMAGES
    path = f"{path_dataset}/test_pngs/791f2ca1aa5b2dc6676d37982b3dc354.png"
    # path = f"{path_dataset}/test_pngs/3ac504655919e5c97111b8d644209ff7.png"
    
    label = test_csv[test_csv["image_id"] == path.split("/")[-1].split(".")[0]]["abnormal"].values[0]
    image_224 = np.array(Image.open(path).resize((224, 224)))
    data = np.repeat(image_224[..., np.newaxis], 3, axis=-1)
    image_224 = transforms.ToTensor()(data)
    image_224 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(image_224)
    image_224 = torch.unsqueeze(image_224, 0)
    probs_abnormal = classifier(image_224)[0, 1].detach().numpy()

    print("probability of the object is abnormal:" ,probs_abnormal , "while the ground truth is", label)

    

