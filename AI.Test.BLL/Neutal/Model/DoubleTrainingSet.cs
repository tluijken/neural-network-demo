namespace AI.Test.BLL.Neutal.Model
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    public class DoubleTrainingSet
    {
        public double Tollerance { get; set; }

        public double[][] InputSet { get; set; }

        public double[][] OutputSet { get; set; }

        public int IterationsPerRun { get; set; }

        public List<List<double>> CalculatedOutputSet { get; set; } = new List<List<double>>();

        public bool Finished
        {
            get
            {
                if (CalculatedOutputSet == null || !CalculatedOutputSet.Any())
                {
                    return false;
                }
                // Check if the set has calculated all the outputs for the entire set correctly
                for (var i = 0; i < OutputSet.Length; i++)
                {
                    for (var j = 0; j < OutputSet[i].Length; j++)
                    {
                        if (Math.Abs(CalculatedOutputSet[i][j] - OutputSet[i][j]) > Tollerance)
                        {
                            return false;
                        }
                    }
                }
                return true;
            }
        }
    }
}