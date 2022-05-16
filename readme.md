# Lung Diseases Classification on tf.js and tf.node

Node.js repo for training conv neural network and predicting on loaded image

## Installation

You need to install node modules and dependencies

    npm install

## Training

For training, I used this [dataset](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis).
If you want to train the model, you have to create folder `data` and then add some data there (be sure to divide it by classes).
I used it only with 4 classes from dataset, but it can train with other amount. To run training:

    npm run train

Warning: by default train module uses GPU with CUDA library, so if you don't have one you have to replace every tfnode import from `@tensorflow/tfjs-node-gpu` to `@tensorflow/tfjs-node`.

## Predicting

To use trained model for prediction run:

    npm run predict