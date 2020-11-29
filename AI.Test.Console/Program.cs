using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AI.Test.BLL.Neutal.Configuration;
using AI.Test.BLL.Neutal.Model;

namespace AI.Test.Demo
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            const double high = .99;
            const double low = .1;

            var doubleTrainingSet = new DoubleTrainingSet
            {
                InputSet = new[]
                {
                    new[] { high, high},
                    new[] { low, high},
                    new[] { high, low},
                    new[] { low, low}
                },
                OutputSet = new[]
                {
                    new[] { low },
                    new[] { high },
                    new[] { high },
                    new[] { low }
                },
                Tollerance = 0.0000000000001,
                IterationsPerRun = 5
            };

            var configuration = new NeuralNetConfiguration
            {
                HiddenLayerNodes = new Dictionary<int, int>
                {
                    { 1, 6 },
                    { 2, 6 }
                },
                LearningSpeed = 3,
                NrOfInputNeurons = doubleTrainingSet.InputSet.Max(d => d.Length),
                NumberOfOutputNeurons = doubleTrainingSet.OutputSet.Max(d => d.Length)
            };

            var net = new FeedForwardNeuralNet(configuration);

            net.Initialize(1);

            var count = 0;

            while (!doubleTrainingSet.Finished)
            {
                count++;

                net.Train(doubleTrainingSet);

                Console.SetCursorPosition(0, 1);
                foreach (var output in doubleTrainingSet.CalculatedOutputSet)
                {
                    var sb = new StringBuilder();
                    foreach (var d in output)
                    {
                        sb.Append(Math.Round(d, 14) + " ");
                    }
                    Console.WriteLine(sb.ToString());
                }
            }

            Console.WriteLine($"{count * doubleTrainingSet.IterationsPerRun} iterations required for training");

            Console.ReadKey();
        }
    }
}