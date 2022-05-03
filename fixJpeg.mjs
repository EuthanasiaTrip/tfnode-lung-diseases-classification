/**
 * Script for re-converting images in its formats. 
 * Run when tf.node.decodeJpeg cannot decode images
 */


import jimp from 'jimp';
import fs from 'fs';

const folder = process.argv[2];
let count = 0;
const timeStart = new Date();

const content = fs.readdirSync(folder);


content.forEach(file => {
    const folderFileName = folder + '/' + file;
    jimp.read(folderFileName)
        .then(image => {
            fs.unlinkSync(folderFileName);
            console.log("rewriting " + folderFileName);
            image.write(folderFileName, onWriteEnd);
        })
        .catch(err => {
            throw err;
        });
});

function onWriteEnd() {
    count++;
    if (count >= content.length) {
        const totalTime = new Date() - timeStart
        const seconds = Math.floor(totalTime / 1000);
        const minutes = Math.floor(seconds / 60);
        console.log(`All images in folder have been fixed.\nTotal count of images: ${count}\nTime spent: ${minutes} min ${seconds} sec`)
    }
}


