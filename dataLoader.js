const fs = require('fs');
const tf = require('@tensorflow/tfjs-node-gpu');
const logger = require('./logger.js');

const IMAGE_H = 200;
const IMAGE_W = 200;

class DataLoader {
    constructor(){}

    getData(dataType) {
        const dataFolder = `data/${dataType}`;
        let data = [];
        let classes = [];
        let dataLabels = [];
    
        fs.readdirSync(dataFolder).forEach(file => {
            classes.push(file);
        });
    
        classes.forEach((label, index) => {
            const folder = dataFolder + '/' + label;
            logger.log(`Resizing and preparing array of 3D tensors. Class: ${label}`);
            fs.readdirSync(folder).forEach(img => {
                let imageTensor = tf.node.decodeJpeg(fs.readFileSync(folder + '/' + img), 1);
                data.push(
                    tf.image.resizeBilinear(imageTensor, [IMAGE_H, IMAGE_W])
                );
                imageTensor.dispose();
                dataLabels.push(index);                              
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
}

module.exports = {IMAGE_H, IMAGE_W, loader: new DataLoader()};