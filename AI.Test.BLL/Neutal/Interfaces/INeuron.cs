namespace AI.Test.BLL.Neutal.Interfaces
{
    using Model;

    public interface INeuron : INeuronSignal, INeuronReceptor
    {
        void Pulse();
        void ApplyLearning(ref double learningRate);
        void InitializeLearning();

        NeuralFactor Bias { get; }

        double Error { get; set; }
    }
}