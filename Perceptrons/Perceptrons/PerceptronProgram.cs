using System;

namespace Perceptrons
{
    class PerceptronProgram
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin perceptron demo\n"); 
            Console.WriteLine("Predict liberal (-1) or conservative (+1) from age, income"); 

            // Create and train perceptron. 

            double[][] trainData = new double[8][];
            trainData[0] = new double[] { 1.5, 2.0, -1 };
            trainData[1] = new double[] { 2.0, 3.5, -1 };
            trainData[2] = new double[] { 3.0, 5.0, -1 };
            trainData[3] = new double[] { 3.5, 2.5, -1 }; 
            trainData[4] = new double[] { 4.5, 5.0,  1  };
            trainData[5] = new double[] { 5.0, 7.0,  1  }; 
            trainData[6] = new double[] { 5.5, 8.0,  1  }; 
            trainData[7] = new double[] { 6.0, 6.0,  1  };

            Console.WriteLine("The training data is \n");
            ShowData(trainData);
            Console.WriteLine("\n Creating Perceptron");

            int numInput = 2;
            Perceptron p = new Perceptron(numInput);

            double alpha = 0.001;
            int maxEpochs = 100;
            Console.Write("\n Setting learning rate to " + alpha.ToString("F3"));
            Console.WriteLine(" and maxEpochs to " + maxEpochs);

            Console.WriteLine("\n Begin Training");
            double[] weights = p.Train(trainData, alpha, maxEpochs);
            Console.WriteLine("Training Complete");

            Console.WriteLine("\n Best weigths and bias found:");
            ShowVector(weights, 4, true);

            double[][] newData = new double[6][];
            newData[0] = new double[] { 3.0, 4.0 }; //should be -1.
            newData[1] = new double[] { 0.0, 1.0 }; // should be -1.
            newData[2] = new double[] { 2.0, 5.0 }; //should be -1.
            newData[3] = new double[] { 5.0, 6.0 }; //should be 1.
            newData[4] = new double[] { 9.0, 9.0 }; // Should be 1.
            newData[5] = new double[] { 4.0, 6.0 }; // Should be 1.

            Console.WriteLine("\n Predictions for new people: \n");
            for (int i = 0; i < newData.Length; ++i)
            {
                Console.Write("Age, Income = ");
                ShowVector(newData[i], 1, false);

                int c = p.ComputeOutput(newData[i]);
                Console.Write(" Prediction is ");
                if (c == -1)
                    Console.WriteLine("(-1) liberal ");
                else if (c == 1)
                    Console.WriteLine("(+1) conservative");
            }

                Console.WriteLine("\nEnd perceptron demo\n");
            Console.ReadLine();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="trainData"></param>
        static void ShowData(double[][] trainData)
        {
            int numRows = trainData.Length;
            int numCols = trainData[0].Length;
            for (int i = 0; i < numRows; ++i)
            {
                Console.Write("[" + i.ToString().PadLeft(2, ' ') + "] ");
                for (int j = 0; j < numCols - 1; ++j)
                    Console.Write(trainData[i][j].ToString("F1").PadLeft(6));
                Console.WriteLine(" -> " + trainData[i][numCols - 1].ToString("+0;-0"));
            }

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="vector"></param>
        /// <param name="decimals"></param>
        /// <param name="newLine"></param>
        static void ShowVector(double[] vector, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] >= 0.0)
                    Console.Write(" "); //For sign.
                Console.Write(vector[i].ToString("F" + decimals) + " ");
            }

            if (newLine == true)
                Console.WriteLine("");
        }

        
    }

    public class Perceptron
    {
        /// <summary>
        /// Holds the number of x-data features(properties)
        /// </summary>
        private int numInput;
        /// <summary>
        /// Holds the values of x-data.
        /// </summary>
        private double[] inputs;
        /// <summary>
        /// Holds the values of the weights associate with each input 
        /// value both during and after training.
        /// </summary>
        private double[] weights;
        /// <summary>
        /// Holds the value added during the computation of the 
        /// perceptron output.
        /// </summary>
        private double bias;
        /// <summary>
        /// Holds the computed value of the perceptron.
        /// </summary>
        private int output;
        /// <summary>
        /// Random Object used in perceptron constructor &
        /// during the training process.
        /// </summary>
        private Random rnd;

        /// <summary>
        /// Constructor for the percetron
        /// </summary>
        /// <param name="numInput">The number of inputs</param>
        public Perceptron(int numInput)
        {
            this.numInput = numInput;
            this.inputs = new double[numInput];
            this.weights = new double[numInput];
            this.rnd = new Random(0);
            InitialiseWeights();
        }

        /// <summary>
        /// Initialises the weights and defines a value for the bias constant.
        /// </summary>
        private void InitialiseWeights()
        {
            double lo = -0.01;
            double hi = 0.01;
            for (int i = 0; i < weights.Length; ++i)
                weights[i] = (hi - lo) * rnd.NextDouble() + lo;
            bias = (hi - lo) * rnd.NextDouble() + lo;
        }

       

        /// <summary>
        /// checks wether the argument(v) is greater or equal to 0.0
        /// If yes returns +1 Else Returns -1
        /// </summary>
        /// <param name="v">Value to apply Activation on.</param>
        /// <returns></returns>
        private static int Activation(double v)
        {
            if (v >= 0.0)
                return +1;
            else
                return -1;
        }

        /// <summary>
        /// Iteratively adjusts the weights and bias values so that the computed outputs 
        /// for a given set of training data x-values closely match the known outputs.
        /// </summary>
        /// <param name="trainData">A matrix of training data</param>
        /// <param name="alpha">A learning rate.</param>
        /// <param name="maxEpochs">The loop limit.</param>
        /// <returns></returns>
        public double[] Train(double[][] trainData, double alpha, int maxEpochs)
        {
            int epoch = 0;
            double[] xValues = new double[numInput];
            int desired = 0;

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;
            while (epoch < maxEpochs)
            {
                Shuffle(sequence);
                for (int i = 0; i < trainData.Length; ++i)
                {
                    int idx = sequence[i];
                    Array.Copy(trainData[idx], xValues, numInput);
                    desired = (int)trainData[idx][numInput]; //-1 or +1.
                    int computed = ComputeOutput(xValues);
                    Update(computed, desired, alpha); // Modify weights and bias values
                } // for each data.
                ++epoch;
            }

            double[] result = new double[numInput + 1];
            Array.Copy(this.weights, result, numInput);
            result[result.Length - 1] = bias; // Last cell.
            return result;
        }

        /// <summary>
        /// Shuffles an array using the Fisher-Yates Algorithm
        /// </summary>
        /// <param name="sequence">The array to be shuffled.</param>
        private void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }

        /// <summary>
        ///  Uses the perceptron's weights and bias values to generate the perceptron output.
        ///  Computes a sum of the products of each input and its associated weight, adds the bias value
        ///  and then applies the Activation function.
        /// </summary>
        /// <param name="xValues">An array of input values.</param>
        /// <returns>Computed output.</returns>
        public int ComputeOutput(double[] xValues)
        {
            if (xValues.Length != numInput)
                throw new Exception("Bad xValues in  ComputeOutput");
            for (int i = 0; i < xValues.Length; ++i)
                this.inputs[i] = xValues[i];
            double sum = 0.0;
            for (int i = 0; i < numInput; ++i)
                sum += this.inputs[i] * this.weights[i];
            sum += this.bias;

            int result = Activation(sum);
            this.output = result;
            return result;
        }

        /// <summary>
        /// Calculates the difference between the computed output and the desired output
        /// Stores the difference into a variable named delta.
        /// If computed output is too large, the weight is reduced by an amount (alpha * delta * input[i])
        /// If computed output is too small, the weight is increased by an amount (alpha * delta * input[i])
        /// </summary>
        /// <param name="computed">The computed output value.</param>
        /// <param name="desired">The desired output value.</param>
        /// <param name="alpha">The learning rate.</param>
        private void Update(int computed, int desired, double alpha)
        {
            if (computed == desired) return; // we're god

            int delta = computed - desired;  // if computed > desired, delta is +.

            for(int i=0; i < this.weights.Length; ++i)// Each input-weight pair.
            {
                if (computed > desired && inputs[i] >= 0.0)// need to reduce weights.
                    weights[i] = weights[i] - (alpha * delta * inputs[i]);
                else if (computed > desired && inputs[i] < 0.0)// Need to reduce weights.
                    weights[i] = weights[i] + (alpha * delta * inputs[i]);
                else if (computed < desired && inputs[i] >= 0.0)// Need to increase weights
                    weights[i] = weights[i] - (alpha * delta * inputs[i]);
                else if (computed < desired && inputs[i] < 0.0) // Need to increase weights.
                    weights[i] = weights[i] + (alpha * delta * inputs[i]);

            }// Each weight.

            bias = bias - (alpha * delta);
        }
    }
}
