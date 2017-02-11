using System;

namespace BackProp
{
    class BackPropProgram
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin back-propagation demo\n"); 
            Console.WriteLine("Creating a 3-4-2 neural network\n"); 
            int numInput = 3;
            int numHidden = 4;
            int numOutput = 2;
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput); 

            double[] weights = new double[26] { 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26 };
            Console.WriteLine("Setting dummy initial weights to:");
            
            ShowVector(weights, 8, 2, true);
            nn.SetWeights(weights); 
            double[] xValues = new double[3] { 1.0, 2.0, 3.0 }; // Inputs.
            double[] tValues = new double[2] { 0.2500, 0.7500 }; // Target outputs.

            Console.WriteLine("\nSetting fixed inputs = ");
            ShowVector(xValues, 3, 1, true);

            Console.WriteLine("Setting fixed target outputs = "); 
            ShowVector(tValues, 2, 4, true); 
            double learnRate = 0.05;
            double momentum = 0.01;
            int maxEpochs = 1000;
            Console.WriteLine("\nSetting learning rate = " + learnRate.ToString("F2"));
            Console.WriteLine("Setting momentum = " + momentum.ToString("F2"));
            Console.WriteLine("Setting max epochs = " + maxEpochs + "\n"); 

            nn.FindWeights(tValues, xValues, learnRate, momentum, maxEpochs);
            double[] bestWeights = nn.GetWeights(); 
            Console.WriteLine("\nBest weights found:");
            ShowVector(bestWeights, 8, 4, true);
            Console.WriteLine("\nEnd back-propagation demo\n"); 
            Console.ReadLine();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="vector"></param>
        /// <param name="valsPerRow"></param>
        /// <param name="decimals"></param>
        /// <param name="newLine"></param>
        public static void ShowVector(double[] vector,
            int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % valsPerRow == 0)
                    Console.WriteLine("");
                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }

            if (newLine == true)
                Console.WriteLine("");
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="decimals"></param>
        public static void ShowMatrix(double[][] matrix, int decimals)
        {
            int cols = matrix[0].Length;
            for (int i = 0; i < matrix.Length; ++i)
                ShowVector(matrix[i], cols, decimals, true);
        }


    }
}
