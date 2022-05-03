import fs from 'fs';
import * as tf from '@tensorflow/tfjs-node-gpu';

//const logger = require("./logger.js");

import logger from './logger.js'

const IMAGE_H = 200;
const IMAGE_W = 200;
const batchSize = 24;
const trainEpochs = 15;
const validationSplit = 0.15;

//const classes = ['covid', 'normal', 'tubic', 'pneumo']

function getNumberByLabel(label, classes) {
    let num = 0;
    switch (label) {
        case classes[0]:
            num = 0;
            break;
        case classes[1]:
            num = 1;
            break;
        case classes[2]:
            num = 2; 
            break;
        case classes[3]:
            num = 3;
            break;
    }
    return num;
}

function getData(dataType) {
    const dataFolder = `data/${dataType}`;
    let data = [];
    let classes = [];
    let dataLabels = [];

    fs.readdirSync(dataFolder).forEach(file => {
        classes.push(file);
    });

    classes.forEach(label => {
        const folder = dataFolder + '/' + label;
        logger.log(`Resizing and preparing array of 3D tensors. Class: ${label}`);
        fs.readdirSync(folder).forEach(img => {
            let imageTensor = tf.node.decodeJpeg(fs.readFileSync(folder + '/' + img), 1);
            data.push(
                tf.image.resizeBilinear(imageTensor, [IMAGE_H, IMAGE_W])
            );
            imageTensor.dispose();
            dataLabels.push(getNumberByLabel(label, classes));
        });
    });
    logger.log("Stacking 3D tensors into 4D tensor...");
    const images = tf.stack(data);
    const labels = tf.oneHot(tf.tensor1d(dataLabels, 'int32'), classes.length).toFloat();


    data.forEach(element =>
        element.dispose()
    );

    logger.log(tf.memory(), false);
    return { images, labels }
}

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_H, IMAGE_W, 1],
        kernelSize: [5, 5],
        filters: 128,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [3, 3] }));
    model.add(tf.layers.conv2d({ kernelSize: [5, 5], filters: 64, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: [3, 3] }));
    model.add(tf.layers.conv2d({ kernelSize: [3, 3], filters: 30, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    model.add(tf.layers.conv2d({ kernelSize: [3, 3], filters: 30, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 2048, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.1 }));
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.1 }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}


//const trainData = getData("train");

// trainData.labels.print((message, params) => {
//     logger.log(message, false);
// });

let model = createModel();
model.summary((message) => {
    logger.log(message, false);
});


model.summary();

// trainData.labels.print((message, params) => {
//     logger.log(message, false);
// });


// await model.fit(trainData.images, trainData.labels, {
//     batchSize,
//     validationSplit,
//     epochs: trainEpochs,
//     callbacks: {
//         onBatchEnd: async (batch, logs) => {
//             //logger.log(JSON.stringify(logs));
//         }
//     }
// });

// trainData.images.dispose();
// trainData.labels.dispose();

// const testData = getData("test");
// const evalOutput = model.evaluate(testData.images, testData.labels);
// logger.log(
//     `\nEvaluation result:\n` +
//     `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
//     `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);
