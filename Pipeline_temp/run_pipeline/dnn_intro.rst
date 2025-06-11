DNN Model Training
======================


Model architecture
----------------------

- Recognizing the distinct transcriptomic profiles between cortical and subcortical regions and the higher functional differentiation in human cortical regions, we trained a detached deep neural network model.
- **Cortical model** took separately normalized data of the cortex as input, with an input layer dimension of 4,542 and an output layer of 105 cortical region labels (for BN atlas).
- **Subcortical model** took separately normalized data of the subcortex as input, with an input layer dimension of 5,063 and an output layer of 22 subcortical region labels.
- Three layer network, with number of hidden units: 500, 500, 500.

Training strategy
---------------------

- `cross-entropy loss function <https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`_
- `AdamW optimizer <https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html>`_ with weight decay = 1e-6
- Initial learning rate = 6e-5, with a optimization strategy of `StepLR <https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html>`_
- Maximum epochs = 50, with early stopping strategy
- For each generated fused dataset, we repeated the model training 10 times

Region-specific embeddings
------------------------------

- We fed the regional average expression matrices into the model and extracted the outputs from the final hidden layer, which served as the foundation for constructing the homologous mapping relationships.
