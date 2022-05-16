import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs'
import inquirer from 'inquirer'

const modelsDir = "./model"
const models = fs.readdirSync(modelsDir);
const classes = ['COVID', 'Normal', 'Pneumonia', 'Tuberculosis'];

inquirer
    .prompt([
        {
            type: 'list',
            name: 'model',
            message: 'Choose the trained model: ',
            choices: models
        }
    ])
    .then((answer) => {
        console.log(answer);
        inquirer
            .prompt([
                {
                    type: 'input',
                    name: 'imgPath',
                    message: 'Insert full path to an image: '
                }
            ])
            .then((img) => {
                loadModel(answer.model, (model) => {
                    const imgArray = tf.node.decodeJpeg(fs.readFileSync(img.imgPath), 1);
                    const imgTensor = tf.image.resizeBilinear(imgArray, [200, 200]).expandDims(0);
                    const prediction = model.predict(imgTensor);
                    prediction.print();
                    const classIndex = Array.from(prediction.argMax(1).dataSync());
                    console.log("Most probably it is " + classes[classIndex]);
                });
            });

    })

async function loadModel(modelFolder, callback) {
    const handler = tf.io.fileSystem(`./model/${modelFolder}/model.json`);
    const model = await tf.loadLayersModel(handler);
    callback(model);
}
