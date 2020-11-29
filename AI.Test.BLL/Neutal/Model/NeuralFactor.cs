namespace AI.Test.BLL.Neutal.Model
{
    public class NeuralFactor
    {
        #region Constructors

        public NeuralFactor(double weight)
        {
            Weight = weight;
            Delta = 0;
        }

        #endregion

        #region Member Variables

        #endregion

        #region Properties

        public double Weight { get; set; }

        public double Delta { get; set; }

        #endregion

        #region Methods

        public void ApplyWeightChange(ref double learningRate)
        {
            Weight += Delta * learningRate;
        }

        public void ResetWeightChange()
        {
            Delta = 0;
        }

        #endregion
    }
}