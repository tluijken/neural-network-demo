namespace AI.Test.BLL.Neutal.Model
{
    using System.Collections;
    using System.Collections.Generic;
    using Interfaces;

    public class NeuralLayer : INeuralLayer
    {
        #region Member Variables

        private readonly List<INeuron> _neurons;

        #endregion

        #region Constructor

        public NeuralLayer()
        {
            _neurons = new List<INeuron>();
        }
        
        #endregion

        #region IEnumerable<INeuron> Members

        public IEnumerator<INeuron> GetEnumerator()
        {
            return _neurons.GetEnumerator();
        }

        #endregion

        #region IEnumerable Members

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        #endregion

        #region IList<INeuron> Members

        public int IndexOf(INeuron item)
        {
            return _neurons.IndexOf(item);
        }

        public void Insert(int index, INeuron item)
        {
            _neurons.Insert(index, item);
        }

        public void RemoveAt(int index)
        {
            _neurons.RemoveAt(index);
        }

        public INeuron this[int index]
        {
            get => _neurons[index];
            set => _neurons[index] = value;
        }

        #endregion

        #region ICollection<INeuron> Members

        public void Add(INeuron item)
        {
            _neurons.Add(item);
        }

        public void Clear()
        {
            _neurons.Clear();
        }

        public bool Contains(INeuron item)
        {
            return _neurons.Contains(item);
        }

        public void CopyTo(INeuron[] array, int arrayIndex)
        {
            _neurons.CopyTo(array, arrayIndex);
        }

        public int Count => _neurons.Count;

        public bool IsReadOnly => false;

        public bool Remove(INeuron item)
        {
            return _neurons.Remove(item);
        }

        #endregion

        #region INeuralLayer Members

        public INeuralLayer InputLayer { get; set; }
        public INeuralLayer OutputLayer { get; set; }
        public string Name { get; set; }

        public void Pulse()
        {
            foreach (var n in _neurons)
            {
                n.Pulse();
            }
        }

        public void ApplyLearning(INeuralNet net)
        {
            var learningRate = net.NetConfiguration.LearningSpeed;

            foreach (var n in _neurons)
            {
                n.ApplyLearning(ref learningRate);
            }
        }

        public void InitializeLearning()
        {
            foreach (var n in _neurons)
            {
                n.InitializeLearning();
            }
        }

        #endregion
    }
}