const MODEL_PATH = 
'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json';
let model = undefined;

async function loadModel() {
    model = await tf.loadLayersModel(MODEL_PATH);
    model.summary();

    // Save to local storage for offline access!
    await model.save('localstorage://demo/price-predict-model');
    console.log(JSON.stringify(await tf.io.listModels()));

    // Create a batch of 1.
    const input = tf.tensor2d([[870]]);
    console.log('single', input);

    // Create a batch of 3.
    const inputBatch = tf.tensor2d([[500], [1100], [970]]);
    console.log('batch', inputBatch);

    // Actually make the predictions for each batch.
    const result = model.predict(input);
    const resultBatch = model.predict(inputBatch);

    const resultArray = result.array();
    console.log(resultArray);
    const resultBatchArray = resultBatch.array();
    console.log(resultBatchArray);

    // Clean up to prevent memory leakage
    input.dispose();
    inputBatch.dispose();
    result.dispose();
    resultBatch.dispose();
    model.dispose();
};

loadModel();