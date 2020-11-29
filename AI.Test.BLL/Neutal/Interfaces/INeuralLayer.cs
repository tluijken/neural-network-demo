namespace AI.Test.BLL.Neutal.Interfaces
{
    using System.Collections.Generic;

    /// <summary>
    ///     A neural layer is a member of the NeuralNet containing a number of neurons.
    ///     A neural layer can be either the input layer, a hidden layer or the outputlayer of the neural net. 
    /// </summary>
    public interface INeuralLayer : IList<INeuron>
    {
        /// <summary>
        ///     Gets or sets the input layer for this neural layer.
        ///     For the INPUT layer in the neural net this value is NULL.
        /// </summary>
        INeuralLayer InputLayer { get; set; }

        /// <summary>
        ///     Gets or sets the output layer for this neural layer.
        ///     For the OUTPUT layer in the neural net this value is NULL
        /// </summary>
        INeuralLayer OutputLayer { get; set; }

        /// <summary>
        ///     Gets the name of the neural layer
        /// </summary>
        string Name { get; }

        /// <summary>
        ///     Pulses the input signal through the nodes in the layer
        /// </summary>
        void Pulse();

        /// <summary>
        ///     Applies learning for this layer.
        /// </summary>
        /// <param name="net">The neuralnet this layer is a member of.</param>
        void ApplyLearning(INeuralNet net);

        /// <summary>
        ///     Initializes all the nodes and connections in this layer for learning.
        /// </summary>
        void InitializeLearning();
    }
}