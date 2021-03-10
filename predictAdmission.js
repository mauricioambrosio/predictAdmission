// install libraries below before running
const csv = require('csv-parser');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const _ = require('underscore');


const ignoreCol = [
    'Serial No.', 
    //'GRE Score', 
    //'TOEFL Score',
    //'University Rating',
    //'SOP',
    //'LOR',
    //'CGPA',
    //'Research'
]

// get dataset from fileName
function getDataset(fileName){
    const dataset = []; 
    return new Promise((resolve, reject)=>{
        fs.createReadStream(fileName)
            .pipe(csv())
            .on('data', (row) => {
                dataset.push(row)
            })
            .on('end', () => {
                console.log('Done!');
                resolve(dataset);
            })
            .on('error', ()=>{
                resolve([]);
            })
    });
}

function compileModel(n_cols){
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units:28,
        inputShape:[n_cols],
        activation:'sigmoid'
    }));

    model.add(tf.layers.dense({
        units:28,
        activation:'sigmoid'
    }));
    
    model.add(tf.layers.dense({
        units:1,
        activation:'sigmoid'
    }));

    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'adam',
    })
    
    return model;
}

// transform dataset from array of dictionaries to 2d array and return training and testing datasets
async function processDataset(dataset){
    console.log('dataset.length', dataset.length);
    
    datasetShuffled = _.shuffle(dataset);

    // the model will only consider the columns that are commented
       
    const label = 'Chance of Admit ';

    X = datasetShuffled.map(row => {
        const rowArray = [];
        Object.keys(row).forEach((key)=>{
            if(!(ignoreCol.includes(key.trim())) && key !== label ){
                rowArray.push(parseFloat(row[key]));
            }
        });
        return rowArray;
    });

    Y = datasetShuffled.map(row => {
        return parseFloat(row[label]);
    })

    let X_train = X.splice(0,400);
    let Y_train = Y.splice(0,400);

    let X_test = X.splice(0,100);
    let Y_test = Y.splice(0,100);
    return {X_train, Y_train, X_test, Y_test};
}

function scaleMinMax(value, newMin, newMax, min, max ){
    return newMin + ( (value - min) * (newMax - newMin) ) / (max - min);
}

// normalize using min max strategy
function normalize(X, newMin, newMax, params){
    const {mins, maxs} = params;
    const n_rows = X.length;
    const n_cols = X[0].length;
    
    if(mins.length != n_cols || maxs.length != n_cols){
        return console.log("wrong dimensions.");
    } 
    let i,j;
    for(i=0;i<n_rows;i++){
        for(j=0;j<n_cols;j++){
            X[i][j] = scaleMinMax(X[i][j], newMin, newMax, mins[j], maxs[j]);
        }
    }
}

function getMinMax(X){
    n_rows = X.length;
    n_cols = X[0].length;

    let mins = new Array(n_cols);
    mins.fill(Infinity);
    let maxs = new Array(n_cols);
    maxs.fill(-Infinity);

    let i,j;
    for(i=0;i<n_rows;i++){
        for(j=0;j<n_cols;j++){
            let val = X[i][j];
            if(val<mins[j]) mins[j] = val;
            if(val>maxs[j]) maxs[j] = val;
        }
    }
    return {mins,maxs};
}


async function trainModel(model, epochs, X_train_tf, Y_train_tf, X_test_tf, Y_test_tf){      
    return new Promise((resolve, reject)=>{
        model.fit(X_train_tf, Y_train_tf, {
            epochs: epochs,
            validationData: [X_test_tf, Y_test_tf],
        }).then(()=>{
            console.log('model trained.');
            resolve(model);
        }).catch((error)=>{
            console.log(error.message);
        });
    })
}

function testModel(model, X_test, Y_test){
    X_test_tf = tf.tensor2d(X_test);
    let P_tf = model.predict(X_test_tf);
    P = P_tf.dataSync();

    for(i=0;i<Y_test.length;i++){
        console.log(Y_test[i],'<->', P[i]);
    }
}

async function run(){
    let dataset = await getDataset('Admission_Predict_Ver1.1.csv');
    let datasetProcessed = await processDataset(dataset);

    X_train = datasetProcessed.X_train;
    Y_train = datasetProcessed.Y_train;
    X_test = datasetProcessed.X_test;
    Y_test = datasetProcessed.Y_test;
    
    // normalize datasets to be between -1 and 1
    const paramsMinMax = getMinMax(X_train);
    normalize(X_train, -1, 1, paramsMinMax);
    normalize(X_test, -1, 1, paramsMinMax);

    X_train_tf = tf.tensor2d(X_train);
    Y_train_tf = tf.tensor1d(Y_train);
    X_test_tf = tf.tensor2d(X_test);
    Y_test_tf = tf.tensor1d(Y_test);

    const n_cols = X_train[0].length;
    let model = compileModel(n_cols);

    const epochs = 5000;
    model = await trainModel(model, epochs, X_train_tf, Y_train_tf, X_test_tf, Y_test_tf);

    let X_test_sample = X_test.splice(0,10);
    let Y_test_sample = Y_test.splice(0,10);

    testModel(model, X_test_sample, Y_test_sample);   
    
    model.save('file://./saved_model');
}
run();
