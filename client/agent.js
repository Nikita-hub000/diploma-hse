export const createPongAgent = (modelPath = './dqn/dqn-mid.onnx') => {
  console.log(modelPath);
  let ortSession = null;

  const stateDim = 5;
  const actionDim = 3;

  const loadModel = async () => {
    console.log(`Attempting to load ONNX model from: ${modelPath}`);
    try {
      ortSession = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      console.log('ONNX Runtime session created successfully.');

      if (
        ortSession.inputNames.length === 0 ||
        ortSession.outputNames.length === 0
      ) {
        console.error('Model loaded but has no input or output names defined.');
        ortSession = null;
        return false;
      }
      console.log('Input names:', ortSession.inputNames);
      console.log('Output names:', ortSession.outputNames);
      return true;
    } catch (e) {
      console.error(`Failed to load ONNX model: ${e}`);
      ortSession = null;
      return false;
    }
  };

  const selectAction = async (state) => {
    if (!ortSession) {
      console.error('ONNX session not loaded. Returning default action (0).');
      return 0;
    }
    if (!state || state.length !== stateDim) {
      console.error(
        `Invalid state: expected length ${stateDim}, got ${
          state?.length ?? 'undefined'
        }`
      );
      return 0;
    }

    try {
      const inputTensor = new ort.Tensor(
        'float32',
        state instanceof Float32Array ? state : new Float32Array(state),
        [1, stateDim]
      );

      const feeds = { [ortSession.inputNames[0]]: inputTensor };
      const output = await ortSession.run(feeds);
      const qValues = output[ortSession.outputNames[0]].data;

      return qValues.reduce(
        (bestIdx, val, i) => (val > qValues[bestIdx] ? i : bestIdx),
        0
      );
    } catch (e) {
      console.error(`Inference error: ${e}`);
      return 0;
    }
  };

  return Object.freeze({
    modelPath,
    stateDim,
    actionDim,
    loadModel,
    selectAction,
  });
};
