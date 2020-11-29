namespace AI.Test.BLL.Neutal.Interfaces
{
    using Configuration;

    public interface INeuralNet
    {
        NeuralNetConfiguration NetConfiguration { get; }
        
    }
}