namespace AI.Test.BLL.Neutal.Interfaces
{
    using System.Collections.Generic;
    using Model;

    /// <summary>
    ///     The neural receptor contains information about 
    ///     the connection input from the inputlayer nodes.
    /// 
    ///     Per signal there is a factor (weight) involved. 
    ///     This weight is updated throughout the learning process.
    ///     Usually the initial factor is a valu between 0 and 1.
    /// </summary>
    public interface INeuronReceptor
    {
        Dictionary<INeuronSignal, NeuralFactor> Input { get; }
    }
}