namespace AI.Test.BLL.Neutal.Model
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Configuration;
    using Interfaces;

    /// <summary>
    ///     This neural net uses back-propagation for learning using either the Sigmoid or Tangent activation function
    ///     The net has an input and and output layer. 
    ///     The number of hiddenlayers can be one or many depending on the given neuralnet configuration
    /// 
    ///     A Feed Forward NeuralNet is generally used for classification.
    ///     For example
    ///     Age | Income | Gender | Religion    | Political preference
    ///     31  | 51.000 |   M    | Catholic    | Republic
    ///     54  | 64.000 |   F    | Protestant  | Democratic
    ///     48  | 43.000 |   M    | Protestant  | Republic
    ///     24  | 24.000 |   F    | Catholic    | Republic
    ///     35  | 46.000 |   M    | Protestant  | ?????
    /// </summary>
    public class FeedForwardNeuralNet : INeuralNet
    {
        private IList<INeuralLayer> _hiddenLayers;
       
        private INeuralLayer _inputLayer;

        private INeuralLayer _outputLayer;

        /// <summary>
        /// Initializes a new instance of <see cref="FeedForwardNeuralNet"/>
        /// </summary>
        /// <param name="netConfiguration">The neural net configuration</param>
        public FeedForwardNeuralNet(NeuralNetConfiguration netConfiguration)
        {
            NetConfiguration = netConfiguration;
        }

        public NeuralNetConfiguration NetConfiguration { get; }

        private void pulse()
        {
            lock (this)
            {
                foreach (var hiddenLayer in _hiddenLayers)
                {
                    hiddenLayer.Pulse();
                }

                _outputLayer.Pulse();
            }
        }

        private void applyLearning()
        {
            lock (this)
            {
                foreach (var hiddenLayer in _hiddenLayers)
                {
                    hiddenLayer.ApplyLearning(this);
                }
                _outputLayer.ApplyLearning(this);
            }
        }

        private void initializeLearning()
        {
            lock (this)
            {
                foreach (var hiddenLayer in _hiddenLayers)
                {
                    hiddenLayer.InitializeLearning();
                }
                _outputLayer.InitializeLearning();
            }
        }

        public void Train(DoubleTrainingSet doubleTrainingSet)
        {
            lock (this)
            {
                for (var i = 0; i < doubleTrainingSet.IterationsPerRun; i++)
                {
                    initializeLearning(); // set all weight changes to zero

                    for (var j = 0; j < doubleTrainingSet.InputSet.Length; j++)
                    {
                        BackPropogation_TrainingSession(doubleTrainingSet.InputSet[j], doubleTrainingSet.OutputSet[j]);
                    }

                    applyLearning(); // apply batch of cumlutive weight changes
                }

                // Apply learning after the whole set and iterations again
                applyLearning();

                // Create new storage for the output values
                doubleTrainingSet.CalculatedOutputSet = new List<List<double>>();

                // Reverse over the input set
                foreach (var inputSet in doubleTrainingSet.InputSet.Reverse())
                {
                    // Set the input nodes
                    preparePerceptionLayerForPulse(inputSet);
                    // pulse
                    pulse();

                    // store the output
                    doubleTrainingSet.CalculatedOutputSet.Add(_outputLayer.Select(outputNeuron => outputNeuron.Output)
                        .ToList());
                }
            }
        }


        public void Initialize(int randomSeed)
        {
            var random = new Random(randomSeed);

            _inputLayer = new NeuralLayer { Name = "InputLayer" };
            _outputLayer = new NeuralLayer { Name = "OutputLayer" };
            _hiddenLayers = new List<INeuralLayer>();
            foreach (var hiddenLayer in NetConfiguration.HiddenLayerNodes)
            {
                _hiddenLayers.Add(new NeuralLayer { Name = $"HiddenLayer{hiddenLayer.Key}" });
            }


            for (var i = 0; i < NetConfiguration.NrOfInputNeurons; i++)
            {
                _inputLayer.Add(new Neuron(0));
            }

            for (var i = 0; i < NetConfiguration.NumberOfOutputNeurons; i++)
            {
                _outputLayer.Add(new Neuron(random.NextDouble()));
            }

            _inputLayer.OutputLayer = _hiddenLayers.First();


            for (var i = 0; i < NetConfiguration.HiddenLayerNodes.Count; i++)
            {
                var nrOfNodesInLayer = NetConfiguration.HiddenLayerNodes.Values.ToArray()[i];
                for (var j = 0; j < nrOfNodesInLayer; j++)
                {
                    _hiddenLayers[i].Add(new Neuron(random.NextDouble()));
                }
            }

            // wire-up the hidden layers
            foreach (var hiddenLayer in _hiddenLayers)
            {
                var index = _hiddenLayers.IndexOf(hiddenLayer);
                hiddenLayer.InputLayer = index == 0 ? _inputLayer : _hiddenLayers[index - 1];
                hiddenLayer.OutputLayer = index < _hiddenLayers.Count - 1 ? _hiddenLayers[index + 1] : _outputLayer;

                foreach (var hiddenNode in hiddenLayer)
                {
                    foreach (var inputNode in hiddenLayer.InputLayer)
                    {
                        hiddenNode.Input.Add(inputNode, new NeuralFactor(random.NextDouble()));
                    }
                }
            }

            _outputLayer.InputLayer = _hiddenLayers.Last();

            // wire-up output layer to hidden layer
            foreach (var outputNode in _outputLayer)
            {
                foreach (var hiddenNode in _outputLayer.InputLayer)
                {
                    outputNode.Input.Add(hiddenNode, new NeuralFactor(random.NextDouble()));
                }
            }
        }

        private void calculateErrors(double[] desiredResults)
        {
            // Calculate output error values 
            for (var i = 0; i < _outputLayer.Count; i++)
            {
                var outputNode = _outputLayer[i];
                var temp = outputNode.Output;

                outputNode.Error = (desiredResults[i] - temp) * sigmoidDerivative(temp); //;*temp * (1.0F - temp); // 
            }


            // Backwards traverse through hidden layers
            var reversedHiddenLayers = _hiddenLayers.Reverse().ToList();

            foreach (var hiddenLayer in reversedHiddenLayers)
            {
                // calculate hidden layer error values
                foreach (var hiddenNode in hiddenLayer)
                {
                    var output = hiddenNode.Output;
                    double error = 0;

                    foreach (var outputNode in hiddenLayer.OutputLayer)
                    {
                        error += outputNode.Error * outputNode.Input[hiddenNode].Weight * sigmoidDerivative(output);
                        /*(1.0F - output);*/ //sigmoidDerivative(temp);
                    }

                    hiddenNode.Error = error;
                }
            }
        }

        private static double sigmoidDerivative(double value)
        {
            return value * (1.0d - value);
        }

        private void preparePerceptionLayerForPulse(double[] input)
        {
            int i;

            if (input.Length != _inputLayer.Count)
            {
                throw new ArgumentException($"Expecting {_inputLayer.Count} inputs for this net");
            }

            // initialize data
            for (i = 0; i < _inputLayer.Count; i++)
            {
                _inputLayer[i].Output = input[i];
            }
        }

        private void calculateAndAppendTransformation()
        {
            // adjust output layer weight change
            var revsedHiddenLayers = _hiddenLayers.Reverse().ToList();
            foreach (var hiddenLayer in revsedHiddenLayers)
            {
                foreach (var outputNode in hiddenLayer.OutputLayer)
                {
                    foreach (var hiddenNode in hiddenLayer)
                    {
                        outputNode.Input[hiddenNode].Delta += outputNode.Error * hiddenNode.Output;
                    }

                    outputNode.Bias.Delta += outputNode.Error * outputNode.Bias.Weight;
                }

                // adjust hidden layer weight change
                foreach (var hiddenNode in hiddenLayer)
                {
                    foreach (var inputNode in hiddenLayer.InputLayer)
                    {
                        hiddenNode.Input[inputNode].Delta += hiddenNode.Error * inputNode.Output;
                    }

                    hiddenNode.Bias.Delta += hiddenNode.Error * hiddenNode.Bias.Weight;
                }
            }
        }


        private void BackPropogation_TrainingSession(double[] input, double[] desiredResult)
        {
            preparePerceptionLayerForPulse(input);
            pulse();
            calculateErrors(desiredResult);
            calculateAndAppendTransformation();
        }
    }
}