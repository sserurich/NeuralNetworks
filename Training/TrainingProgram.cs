using System;

namespace Training
{
    class TrainingProgram
    {
        static void Main(string[] args)
        {
        }

        /// <summary>
        /// Shows a matrix 
        /// </summary>
        /// <param name="matrix">The matrix to be shown.</param>
        /// <param name="numRows">The number of rows for the matrix.</param>
        /// <param name="decimals">The number of decimal places.</param>
        /// <param name="newLine">A boolean value indication whether to display the newline.</param>
        static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool newLine)
        {
            for (int i = 0; i < numRows; ++i)
            {
                Console.Write(i.ToString().PadLeft(3) + ": ");

                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    if (matrix[i][j] >= 0.0) 
                        Console.Write(" "); 
                    else Console.Write("-");
                    Console.Write(Math.Abs(matrix[i][j]).ToString("F" + decimals) + " ");
                }
                Console.WriteLine("");
            }

            if (newLine == true)
                Console.WriteLine("");
        }

        static void MakeTrainTest(double[][] allData, int seed, 
            out double [][] trainData, out double [][]testData){

      }

        static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {

        }
    }
}
