using System;

namespace FeedForward
{
    public class NeuralNetwork
    {
        public enum Activation { HyperTan, LogSigmoid, Softmax };
        private Activation hActivation;
        private Activation oActivation;

        /// <summary>
        /// Stores the number of input nodes of the neural network
        /// </summary>
        private int numInput;

        /// <summary>
        /// Stores the number of hidden nodes.
        /// </summary>
        private int numHidden;

        /// <summary>
        /// Stores the number of output nodes.
        /// </summary>
        private int numOutput;

        /// <summary>
        /// Holds the weights from input nodes to hidden nodes where
        /// row index corresponds to input node & column index corresponds to the index
        /// of the hidden node.
        /// </summary>
        private double[][] ihWeights;

        /// <summary>
        /// Holds the bias values for the hidden nodes
        /// </summary>
        private double[] hBiases;

        /// <summary>
        /// Stores the hidden node outputs after summing the products of weights & inputs 
        /// adding the bias values & applying the activation function.
        /// </summary>
        private double[] hOutputs;

        /// <summary>
        /// Holds the weight from hidden nodes to output nodes.
        /// </summary>
        private double[][] hoWeights;

        /// <summary>
        /// Holds the bias values for the output nodes.
        /// </summary>
        private double[] oBiases;

        /// <summary>
        /// Holds the final overall computed neuralnetwork output values.
        /// </summary>
        private double[] outputs;

        /// <summary>
        /// Holds the numeric inputs to the neural network.
        /// </summary>
        private double[] inputs;


        /// <summary>
        /// 
        /// </summary>
        /// <param name="numInput"></param>
        /// <param name="numHidden"></param>
        /// <param name="numOutput"></param>
        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            this.inputs = new double[numInput];
            this.ihWeights = MakeMatrix(numInput, numHidden);
            this.hBiases = new double[numHidden];
            this.hOutputs = new double[numHidden];

            this.hoWeights = MakeMatrix(numHidden, numOutput);
            this.oBiases = new double[numOutput];
            this.outputs = new double[numOutput];

        }

        /// <summary>
        /// Makes a two  dimension array of type double.
        /// </summary>
        /// <param name="rows">Number of rows for the 2-dimension array to be made.</param>
        /// <param name="cols">Number of cols for the 2-dimension array to be made.</param>
        /// <returns>The Created 2-dimension array of type double.</returns>
        public static double[][] MakeMatrix(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }
        
        /// <summary>
        /// Sets the weights for ihWeights, hoWeights 
        /// and biases for hidden neurons and output neurons.
        /// Assumes values in weights are are stored in a particular order.
        /// </summary>
        /// <param name="weights">An array containing the weights.</param>
        public void SetWeights(double[] weights)
        {
            int numWeights = (numInput * numHidden) + numHidden + 
                (numHidden * numOutput) + numOutput;

            if (weights.Length != numWeights)
                throw new Exception("Bad weights array");

            int k = 0;

            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    ihWeights[i][j] = weights[k++];
            

            for (int i = 0; i < numHidden; ++i)
                hBiases[i] = weights[k++];

            for (int i = 0; i < numHidden; ++i)
                for (int j = 0; j < numOutput; ++j)
                    hoWeights[i][j] = weights[k++];

            for (int i = 0; i < numOutput; ++i)
                oBiases[i] = weights[k++];

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="xValues"></param>
        /// <returns></returns>
        public double[] ComputeOutPuts(double[] xValues)
        {
            if (xValues.Length != numInput)
                throw new Exception("Bad xValues array");

            double[] hSums = new double[numHidden];
            double[] oSums = new double[numOutput];

            for (int i = 0; i < xValues.Length; ++i)
                inputs[i] = xValues[i];

            for (int j = 0; j < numHidden; ++j)
                for (int i = 0; i < numInput; ++i)
                    hSums[j] += inputs[i] * ihWeights[i][j];

            for (int i = 0; i < numHidden; ++i)
                hSums[i] += hBiases[i];

            Console.WriteLine("\n Pre-Activation hidden sums:");
            FeedForwardProgram.ShowVector(hSums, 4, 4, true);

            for (int i = 0; i < numHidden; ++i)
                hOutputs[i] = HyperTan(hSums[i]);

            Console.WriteLine("\nHidden outputs:");
            FeedForwardProgram.ShowVector(hOutputs, 4, 4, true);

            for (int j = 0; j < numOutput; ++j)
                for (int i = 0; i < numHidden; ++i)
                    oSums[j] += hSums[i] * hoWeights[i][j];

            for (int i = 0; i < numOutput; ++i)
                oSums[i] += oBiases[i];

            Console.WriteLine("\nPre-activation output sums:");
            FeedForwardProgram.ShowVector(oSums, 2, 4, true);

            double[] softOut = Softmax(oSums);
            for (int i = 0; i < outputs.Length; ++i)
                outputs[i] = softOut[i];

            double[] result = new double[numOutput];
            for (int i = 0; i < outputs.Length; ++i)
                result[i] = outputs[i];

            return result;
        }


        /// <summary>
        /// Returns the hperbolic tanget of the passed in argument.
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        private static double HyperTan(double v)
        {
            if (v < -20.0)
                return -1.0;
            else if (v > 20.0)
                return 1.0;
            else
                return Math.Tanh(v);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        private static double LogSigmoid(double x)
        {
            if (x < -45.0)
                return 0.0;
            else if (x > 45.0)
                return 1.0;
            else
                return 1.0 / (1.0 + Math.Exp(-x));
        }

        /// <summary>
        /// Implements the softmax activation function.
        /// </summary>
        /// <param name="oSums"></param>
        /// <returns></returns>
        private static double[] Softmax(double[] oSums)
        {
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max)
                    max = oSums[i];

            double scale = 0.0;

            for (int i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale;
            return result;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public double[] GetOutputs()
        {
            double[] result = new double[numOutput];
            for (int i = 0; i < numOutput; ++i)
                result[i] = this.outputs[i];
            return result;
        }


        public static double[] SoftmaxNaive(double[] oSums)
        {
            double denom = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                denom += Math.Exp(oSums[i]);
            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i]) / denom;
            return result;
        }


    }
}
