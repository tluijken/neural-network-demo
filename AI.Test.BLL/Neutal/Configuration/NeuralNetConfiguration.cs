namespace AI.Test.BLL.Neutal.Configuration
{
    using System.Collections.Generic;


    /// <summary>
    ///     A container for the neural net configuration parameters
    /// </summary>
    public class NeuralNetConfiguration
    {
        /// <summary>
        ///     Gets or sets the number of input neutrons
        /// </summary>
        public int NrOfInputNeurons { get; set; }

        /// <summary>
        ///     Gets or sets the number of output neutrons
        /// </summary>
        public int NumberOfOutputNeurons { get; set; }

        /// <summary>
        ///     Gets or sets a key-value pair for the Hidden Layer configuration:
        ///     The number of hidden layers and the number of neurons per hidden layer
        /// </summary>
        public Dictionary<int, int> HiddenLayerNodes { get; set; }
        
        /// <summary>
        ///     Gets or sets the learning speed of the neural net.
        /// </summary>
        public double LearningSpeed { get; set; }
    }
}