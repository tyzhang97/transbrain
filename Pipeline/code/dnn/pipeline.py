# pipeline.py
import os
import numpy as np
from tqdm import tqdm
from config.logger import get_logger 
from models import train_dnn
from dataloader import generate_matrix
from analyzer import transform_human_mouse,independent_test,train_test_split
from sklearn.preprocessing import StandardScaler
 
class DnnPipeline(object):

    def __init__(self, config):
        self.cfg = config
        self._setup_directories()
        self.logger = get_logger()

    def _setup_directories(self):

        os.makedirs(f"{self.cfg.data_files['save_path']}", exist_ok=True)

    def execute(self):

        if self.cfg.train:
            self.logger.info("Starting training")
            self.train()
        if self.cfg.trans:
            self.logger.info("Starting transing")
            self.trans()
        if self.cfg.independent_test:
            self.logger.info("Starting independent testing")
            self.test()

    def train(self):
       
        self.logger.info(f"Training dataset: {self.cfg.data_files['data_path']}")
        self._process_single_dataset(self.cfg.data_files['data_path'])

    def trans(self):

        self.logger.info(f"Performing cross-species transformation")
        transform_human_mouse(self.cfg)

    def test(self):
        
        self.logger.info("Starting independent training")
        self._process_single_dataset(self.cfg.data_files['independent_data_path'])
        self.logger.info("Starting independent testing")
        independent_test(params = self.cfg)

    def _process_single_dataset(self, data_path):

        self.logger.info("Loading dataset")
        X, y,_,sample_id,all_id = generate_matrix(data_path, self.cfg)
        dataset = np.concatenate((X, y), axis=1)
        self.logger.info(f"Dataset shape: {X.shape}")
        
        for repeat_i in tqdm(range(self.cfg.repeat_n)):

            self.logger.info(f"Training iteration {repeat_i+1}/{self.cfg.repeat_n}")
            train, valid = train_test_split(dataset, valid_size=self.cfg.valid_ratio, sample_id = sample_id, all_id = all_id)
            scaler = StandardScaler()
            train_features = train[:, :-1]
            train_features_scaled = scaler.fit_transform(train_features)
            valid_features = valid[:, :-1]
            valid_features_scaled = scaler.transform(valid_features)
            if self.cfg.shuffle:
                train_scaled = np.hstack((train_features_scaled, np.random.permutation(train[:, -1].reshape(-1, 1))))
            else:
                train_scaled = np.hstack((train_features_scaled, train[:, -1].reshape(-1, 1)))
            valid_scaled = np.hstack((valid_features_scaled, valid[:, -1].reshape(-1, 1)))
            self._train_single_model(train_scaled, valid_scaled, repeat_i,data_path)

    def _train_single_model(self, train_data, valid_data, iteration,data_path):

        train_dnn(
            params=self.cfg,
            train_data=train_data,
            valid_data=valid_data,
            iteration=iteration,
            data_path = data_path
        )