namespace AI.Test.BLL.Neutal.Model
{
    using System;
    using System.Collections.Generic;
    using Interfaces;

    public class Neuron : INeuron
    {
        #region Constructors

        public Neuron(double bias)
        {
            Bias = new NeuralFactor(bias);
            Error = 0;
            Input = new Dictionary<INeuronSignal, NeuralFactor>();
        }
        
        public double Output { get; set; }

        #endregion

        #region INeuronReceptor Members

        public Dictionary<INeuronSignal, NeuralFactor> Input { get; }

        #endregion

        #region Private Static Utility Methods

        private static double sigmoid(double value) => 1 / (1 + Math.Exp(-value));

        #endregion

        #region Member Variables

        #endregion

        #region INeuron Members

        public void Pulse()
        {
            lock (this)
            {
                Output = 0;

                foreach (var item in Input)
                {
                    Output += item.Key.Output * item.Value.Weight;
                }

                Output += Bias.Weight;

                Output = sigmoid(Output);
            }
        }

        public NeuralFactor Bias { get; }

        public double Error { get; set; }

        public void ApplyLearning(ref double learningRate)
        {
            foreach (var m in Input)
            {
                m.Value.ApplyWeightChange(ref learningRate);
            }

            Bias.ApplyWeightChange(ref learningRate);
        }

        public void InitializeLearning()
        {
            foreach (var m in Input)
            {
                m.Value.ResetWeightChange();
            }

            Bias.ResetWeightChange();
        }

        #endregion
    }
}